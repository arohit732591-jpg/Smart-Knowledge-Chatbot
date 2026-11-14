import hashlib
import io
import os
import re
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
from bs4 import SoupStrainer
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (WebBaseLoader,PyPDFLoader)
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. PATHS, CONFIGURATION AND PAGE SETUP
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

DB_DIR = BASE_DIR / "chroma_db"
DB_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ENV_FILE)

st.set_page_config(
    page_title="Intelligent RAG Chatbot",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Intelligent RAG-Based Chatbot")
st.subheader("Upload CSV or PDF files, or enter website URLs, and ask questions")


# --------------------------------------------------
# 2. API KEY AND MODELS
# --------------------------------------------------

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Groq API key not found. Check that GROQ_API_KEY is set in your project .env file.")
    st.stop()

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm(groq_api_key):
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=groq_api_key,
    )

embedding_model = load_embedding_model()
llm = load_llm(api_key)

# --------------------------------------------------
# CONVERSATION MEMORY
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []


# --------------------------------------------------
# 3. RAG PROMPT
# --------------------------------------------------

prompt = PromptTemplate(
    input_variables=["context", "question", "history"],
    template="""
You are a helpful knowledge assistant.

Answer only with facts written in the context. Do not add rules,
permissions, prohibitions or penalties that the context does not state.

You may draw a direct conclusion only when the context states a
condition for eligibility or permission. For example, "Members must
hold a valid pass to be eligible to enter the gym" means a member
without a pass is not eligible.

A plain requirement is NOT a prohibition. For example, "Visitors
must sign in at the front desk" does not tell you whether visitors
are allowed after closing time.

If the context only partly answers the question, give the part it
does answer and say the rest is not mentioned in the knowledge base.

Only use a passage if it is about the same subject as the question.
Do not carry a fact over from one subject to another just because
they share a word.

Use conversation history to understand follow-up questions.

If the context has nothing relevant to the question, say exactly:
"Sorry, I couldn't find this information in the uploaded knowledge base."

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""
)

# Prompt for rewriting follow-up questions

rewrite_prompt = PromptTemplate(
    input_variables=["history", "question"],
    template="""
You are a question rewriting assistant.

Your task is to convert the user's latest question
into a standalone question using the conversation history.

Rules:
1. Resolve references such as "it", "that", and "they" using the conversation.
2. If the user says "can't", "cannot", or "don't" after discussing a requirement,
   interpret it as being unable to meet that previously discussed requirement.
3. Preserve the user's intended meaning.
4. Do not answer the question.
5. Do not add unrelated information.
6. If the question is already standalone, return it unchanged.
7. Return only the rewritten question.

Conversation History:
{history}

Latest Question:
{question}

Standalone Question:
"""
)

# --------------------------------------------------
# 4. INPUTS
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Choose one or more CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

# PDF document upload
uploaded_pdfs = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader"
)

st.subheader("Or add website URLs")

website_input = st.text_area(
    "Enter website URLs (one URL per line):",
    placeholder="https://example.com\nhttps://example.org",
)

website_urls = list(
    dict.fromkeys(
        line.strip()
        for line in website_input.splitlines()
        if line.strip()
    )
)


# --------------------------------------------------
# 5. HELPERS: FINGERPRINT, CSV LOADING AND WEBSITE CLEANING
# --------------------------------------------------

def make_knowledge_base_hash(files, pdf_files, urls):
    """Create a stable ID from the selected knowledge sources."""

    digest = hashlib.sha256()

    # Include CSV files
    for uploaded_file in files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())

    # Include PDF files
    for uploaded_pdf in pdf_files:
        digest.update(uploaded_pdf.name.encode("utf-8"))
        digest.update(uploaded_pdf.getvalue())

    # Include website URLs
    for url in sorted(urls):
        digest.update(url.encode("utf-8"))

    return digest.hexdigest()


def clean_website_text(text):
    """Remove blank/repeated lines and common navigation-only text."""
    text = text.replace("\xa0", " ")
    lines = []

    navigation_terms = {
        "home",
        "menu",
        "search",
        "sign in",
        "log in",
        "sign up",
        "contact",
        "privacy",
        "terms",
        "twitter",
        "linkedin",
        "github",
    }

    seen = set()

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()

        if not line:
            continue

        if line.lower() in navigation_terms:
            continue

        # Ignore very short menu-like fragments.
        if len(line) < 3:
            continue

        if line in seen:
            continue

        seen.add(line)
        lines.append(line)

    return "\n".join(lines)


def load_website_document(url):
    """Load a web page, preferring its main/article content."""
    # First try to extract the main article/content area.
    try:
        loader = WebBaseLoader(
            web_path=url,
            bs_kwargs={"parse_only": SoupStrainer(["main", "article"])},
            bs_get_text_kwargs={"separator": "\n"},
        )
        documents = loader.load()
        useful_text = "\n".join(
            doc.page_content for doc in documents if doc.page_content.strip()
        )

        # Some sites do not use main/article tags. Retry with the full page.
        if not useful_text.strip():
            raise ValueError("No main/article content found.")

    except Exception:
        loader = WebBaseLoader(
            web_path=url,
            bs_get_text_kwargs={"separator": "\n"},
        )
        documents = loader.load()
        useful_text = "\n".join(
            doc.page_content for doc in documents if doc.page_content.strip()
        )

    cleaned_text = clean_website_text(useful_text)

    if not cleaned_text:
        raise ValueError("No readable page content was found.")

    return Document(
        page_content=cleaned_text,
        metadata={
            "source": url,
            "source_type": "website",
        },
    )


def load_source_documents(files, pdf_files, urls):
    """Convert CSV rows and web pages into LangChain Documents."""
    documents = []

    # Load uploaded PDF documents
    for uploaded_pdf in pdf_files:

        # Save the uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_pdf.getvalue())
            temp_pdf_path = temp_file.name

        try:
            # Load PDF pages using LangChain
            loader = PyPDFLoader(temp_pdf_path)
            pdf_documents = loader.load()

            # Add source information to each page
            for document in pdf_documents:

                document.metadata["source"] = uploaded_pdf.name
                document.metadata["source_type"] = "pdf"

                documents.append(document)

            st.success(
                f"Successfully loaded PDF: {uploaded_pdf.name}"
            )

        finally:
            # Delete the temporary PDF file
            os.remove(temp_pdf_path)

    for uploaded_file in files:
        dataframe = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))

        st.success(f"Successfully loaded CSV: {uploaded_file.name}")
        st.write(f"Preview of {uploaded_file.name}:")
        st.dataframe(dataframe.head())

        for row_index, row in dataframe.iterrows():
            row_text = "\n".join(
                f"{column}: {value}"
                for column, value in row.items()
                if pd.notna(value)
            )

            if row_text.strip():
                documents.append(
                    Document(
                        page_content=row_text,
                        metadata={
                            "source": uploaded_file.name,
                            "source_type": "csv",
                            "row": int(row_index),
                        },
                    )
                )

    # Important: website loading is outside the CSV loop.
    # Websites therefore work even when no CSV file is uploaded.
    for url in urls:
        st.write(f"Loading website: {url}")
        website_document = load_website_document(url)
        documents.append(website_document)
        st.success(f"Successfully loaded website: {url}")

    return documents


# --------------------------------------------------
# 6. BUILD OR REUSE A PERSISTENT CHROMA DATABASE
# --------------------------------------------------

if uploaded_files or uploaded_pdfs or website_urls:

    try:
        current_hash = make_knowledge_base_hash(uploaded_files, uploaded_pdfs, website_urls)
        collection_name = "kb_" + current_hash[:32]

        st.subheader("Knowledge base")

        # Opening the same collection reuses vectors already saved on disk.
        vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=str(DB_DIR),
        )

        existing_count = vector_db._collection.count()

        if existing_count > 0:
            st.caption("Using the existing persistent ChromaDB. Embeddings were not recreated.")
        else:
            with st.spinner("Loading documents and creating embeddings..."):
                source_documents = load_source_documents(
                    uploaded_files,
                    uploaded_pdfs,
                    website_urls,
                )

                if not source_documents:
                    st.warning("No readable documents were found in the selected sources.")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=120,
                )
                chunks = splitter.split_documents(source_documents)

                if not chunks:
                    st.warning("No text chunks were created from the selected sources.")
                    st.stop()

                # Stable IDs prevent accidental duplicate insertion into this collection.
                chunk_ids = [
                    hashlib.sha256(
                        (
                            chunk.metadata.get("source", "")
                            + str(chunk.metadata.get("row", ""))
                            + str(index)
                            + chunk.page_content
                        ).encode("utf-8")
                    ).hexdigest()
                    for index, chunk in enumerate(chunks)
                ]

                vector_db.add_documents(
                    documents=chunks,
                    ids=chunk_ids,
                )

            st.success("Persistent ChromaDB created successfully!")

        # --------------------------------------------------
        # 7. QUESTION AND ANSWER
        # --------------------------------------------------

        # New question input
        question = st.chat_input(
            "Ask a question about your knowledge base..."
        )

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.latest_sources = []
            st.rerun()  

        
        if question and question.strip():

            # Save the new user question
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            # Convert previous messages into text
            history = "\n".join(
                f"{message['role']}: {message['content']}"
                for message in st.session_state.chat_history[:-1]
            )

            # STEP 1: Rewrite the user's question using conversation history

            with st.spinner("Understanding your question..."):

                rewrite_input = rewrite_prompt.format(
                    history=history,
                    question=question
                )           

                rewrite_response = llm.invoke(rewrite_input)

                retrieval_query = rewrite_response.content.strip()

                # If rewriting returns an empty response, use the original question
                if not retrieval_query:
                    retrieval_query = question

            # STEP 2: Retrieve documents with relevance scores
            with st.spinner("Searching your knowledge base..."):
                scored_results = vector_db.similarity_search_with_relevance_scores(
                    retrieval_query,
                    k=5
                )

            # STEP 3: Set minimum relevance threshold
            MIN_RELEVANCE_SCORE = 0.30

            # STEP 4: Keep only relevant documents
            relevant_results = [
                (doc, score)
                for doc, score in scored_results
                if score >= MIN_RELEVANCE_SCORE
            ]

            # STEP 5: Check whether relevant documents exist
            if not relevant_results:

                fallback_answer = (
                    "Sorry, I couldn't find this information "
                    "in the uploaded knowledge base."
                )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": fallback_answer
                })
                st.session_state.latest_sources = []

            else:

                # STEP 6: Extract relevant documents
                retrieved_docs = [
                    doc for doc, score in relevant_results
                ]

                # STEP 7: Prepare context for the LLM
                context = "\n\n".join(
                    f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                    f"Content: {doc.page_content}"
                    for doc in retrieved_docs
                )

                # STEP 8: Create final prompt
                final_prompt = prompt.format(
                    context=context,
                    question=retrieval_query,
                    history=history
                )

                # STEP 9: Generate AI answer
                with st.spinner("Generating AI answer..."):

                    response = llm.invoke(final_prompt)

                # Save assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.content
                })

                # STEP 10: Save unique sources to show below the answer
                unique_results = []
                seen_sources = set()

                for doc, score in relevant_results:

                    source = doc.metadata.get("source", "Unknown")
                    row = doc.metadata.get("row")

                    # For websites, group chunks by URL.
                    # For CSVs, keep separate rows.
                    source_key = (source, row if row is not None else None)

                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        unique_results.append((doc, score))

                st.session_state.latest_sources = unique_results    

        # Display the complete conversation in the correct order
        st.subheader("Chat")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        # Show sources for the latest answer, directly below it
        if st.session_state.latest_sources:
            st.subheader("Sources")

            for doc, score in st.session_state.latest_sources:
                with st.expander(
                    f"Source: {doc.metadata.get('source', 'Unknown')}"
                ):
                    st.write(doc.page_content)
                    st.caption(f"Relevance score: {score:.3f}")

                    if "row" in doc.metadata:
                        st.caption(f"CSV row: {doc.metadata['row']}")

                    if doc.metadata.get("source_type") == "pdf":
                        page_number = doc.metadata.get("page")
                        if page_number is not None:
                            st.caption(f"PDF page: {page_number + 1}")

                    st.caption(
                        f"Source type: {doc.metadata.get('source_type', 'Unknown')}"
                    )
    except Exception as error:
        st.error(f"An error occurred: {error}")
else:
    st.info("Please upload a CSV or PDF file, or enter at least one website URL to begin.")