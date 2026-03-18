import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

def build_rag_pipeline(doc_path: str):
    # 1. Load document
    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # 3. FREE embeddings via HuggingFace (runs locally)
    print("📦 Loading HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Build FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 5. FREE LLM via Groq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    print("✅ Pipeline ready!\n")
    return llm, retriever


def run_rag(llm, retriever, question: str):
    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]

    # Build prompt
    context_text = "\n\n".join(contexts)
    prompt = f"""Answer the question based only on the following context. Be concise.

Context:
{context_text}

Question: {question}
Answer:"""

    # Call LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content, contexts
