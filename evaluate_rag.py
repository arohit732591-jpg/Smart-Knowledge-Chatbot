import hashlib
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "data" / "college_knowledge_base.csv"
ENV_FILE = BASE_DIR / ".env"

# Match the database-folder logic used in app_backup.py.
PREFERRED_DB_DIR = BASE_DIR / "app" / "chroma_db"
DB_DIR = PREFERRED_DB_DIR if (BASE_DIR / "app").is_dir() else BASE_DIR / "chroma_db"

load_dotenv(ENV_FILE)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"
MIN_RELEVANCE_SCORE = 0.30
TOP_K = 5


# --------------------------------------------------
# 2. EVALUATION QUESTIONS AND REFERENCE ANSWERS
#    All references are based on college_knowledge_base.csv.
# --------------------------------------------------

TEST_CASES = [
    {
        "question": "What is the minimum attendance required in each subject to be eligible for semester examinations?",
        "reference": "Students must maintain a minimum attendance of 80% in each subject to be eligible for semester examinations.",
    },
    {
        "question": "What should students submit if they have a medical emergency affecting attendance?",
        "reference": "Students with medical emergencies must submit valid medical certificates to the academic department.",
    },
    {
        "question": "How many books can a student borrow from the college library at one time?",
        "reference": "Students can borrow a maximum of 4 books at a time using their college identity card.",
    },
    {
        "question": "What are the college library's operating hours and days?",
        "reference": "The college library operates from 9 AM to 5 PM, Monday through Saturday.",
    },
    {
        "question": "Within how many days must library books be returned, and what may happen if they are returned late?",
        "reference": "Books must be returned within 14 days. Late returns may result in a fine.",
    },
    {
        "question": "Which documents must applicants submit during admission?",
        "reference": "Applicants must submit academic transcripts, identity proof, and completed application forms during admission.",
    },
    {
        "question": "When is admission confirmation provided?",
        "reference": "Admission confirmation is provided after document verification and payment of the applicable admission fee.",
    },
    {
        "question": "What minimum CGPA is required to participate in campus placement activities?",
        "reference": "Students must maintain a minimum CGPA of 7.0 to participate in campus placement activities.",
    },
    {
        "question": "What activities does the placement department organize?",
        "reference": "The placement department organizes aptitude training, technical workshops, and mock interviews.",
    },
    {
        "question": "What must students carry to enter the examination hall, and how early must they arrive?",
        "reference": "Students must carry their college identity cards and examination hall tickets and arrive at least 15 minutes before the scheduled examination time.",
    },
]


# --------------------------------------------------
# 3. CHECK REQUIRED FILES AND API KEY
# --------------------------------------------------

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {CSV_FILE.name}. "
        "Place college_knowledge_base.csv in the same folder as evaluate_rag.py."
    )

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY was not found. Check that your project .env file "
        "is in the same folder as evaluate_rag.py."
    )


# --------------------------------------------------
# 4. FIND THE SAME CHROMA COLLECTION AS THE STREAMLIT APP
# --------------------------------------------------

digest = hashlib.sha256()
digest.update(CSV_FILE.name.encode("utf-8"))
digest.update(CSV_FILE.read_bytes())

collection_name = "kb_" + digest.hexdigest()[:32]

if not DB_DIR.exists():
    raise FileNotFoundError(
        f"ChromaDB folder was not found: {DB_DIR}\n"
        "First run app_backup.py and upload college_knowledge_base.csv "
        "so the chatbot creates its persistent knowledge base."
    )

print(f"Using ChromaDB folder: {DB_DIR}")
print(f"Looking for collection: {collection_name}")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_model,
    persist_directory=str(DB_DIR),
)

vector_count = vector_db._collection.count()

if vector_count == 0:
    raise RuntimeError(
        "The matching ChromaDB collection is empty. "
        "Run app_backup.py and upload the original CSV first."
    )

print(f"Connected to ChromaDB. Stored vectors: {vector_count}")


# --------------------------------------------------
# 5. CONNECT TO GROQ
# --------------------------------------------------

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY,
)


# --------------------------------------------------
# 6. RUN THE 10 QUESTIONS THROUGH THE REAL RAG PIPELINE
# --------------------------------------------------

evaluation_rows = []

for number, test_case in enumerate(TEST_CASES, start=1):
    question = test_case["question"]
    reference = test_case["reference"]

    print(f"\nQuestion {number}/{len(TEST_CASES)}: {question}")

    scored_results = vector_db.similarity_search_with_relevance_scores(
        question,
        k=TOP_K,
    )

    relevant_results = [
        (document, score)
        for document, score in scored_results
        if score >= MIN_RELEVANCE_SCORE
    ]

    retrieved_contexts = [
        document.page_content
        for document, score in relevant_results
    ]

    if retrieved_contexts:
        context = "\n\n".join(retrieved_contexts)

        answer_prompt = f"""
You are a helpful college knowledge assistant.
Answer the question using only the supplied context.
Do not invent policies or details.
If the context does not contain the answer, say:
"Sorry, I couldn't find this information in the uploaded knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""
        response = llm.invoke(answer_prompt)
        answer = response.content.strip()
    else:
        answer = (
            "Sorry, I couldn't find this information "
            "in the uploaded knowledge base."
        )

    evaluation_rows.append(
        {
            "user_input": question,
            "retrieved_contexts": retrieved_contexts,
            "response": answer,
            "reference": reference,
        }
    )

    print(f"Retrieved context chunks: {len(retrieved_contexts)}")
    print(f"Generated answer: {answer}")


# --------------------------------------------------
# 7. PREPARE RAGAS METRICS
# --------------------------------------------------

print("\nPreparing RAGAS evaluation...")

evaluation_dataset = EvaluationDataset.from_list(evaluation_rows)

evaluator_llm = LangchainLLMWrapper(llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)

faithfulness_metric = Faithfulness(llm=evaluator_llm)

answer_relevancy_metric = ResponseRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    strictness=1,
)

# --------------------------------------------------
# 8. RUN RAGAS
# --------------------------------------------------

print("Calculating faithfulness and answer relevancy.")
print("This step calls the Groq API and may take a few minutes.")

results = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        faithfulness_metric,
        answer_relevancy_metric,
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    raise_exceptions=True,
)

results_df = results.to_pandas()

# Save detailed results, including questions, answers and metric scores.
output_file = BASE_DIR / "ragas_evaluation_results.csv"
results_df.to_csv(output_file, index=False)

# Save a compact summary as a separate CSV.
score_columns = [
    column
    for column in ["faithfulness", "answer_relevancy"]
    if column in results_df.columns
]

summary = results_df[score_columns].mean(numeric_only=True).to_frame(
    name="average_score"
)
summary_file = BASE_DIR / "ragas_evaluation_summary.csv"
summary.to_csv(summary_file)

print("\n" + "=" * 55)
print("RAGAS EVALUATION COMPLETED")
print("=" * 55)
print("\nAverage metric scores:")
print(summary)
print(f"\nDetailed results saved to: {output_file}")
print(f"Average scores saved to: {summary_file}")
print("\nA score closer to 1.0 is generally better.")
