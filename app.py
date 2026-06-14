import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in your .env file")

print("Loading knowledge base...")
loader = TextLoader("dental_knowledge.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}
    )
except Exception:
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.2)

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a friendly and professional customer support assistant for Neupane Dental Clinic "
        "located in Tilottama-03 Divertole, Rupandehi, Nepal.\n\n"
        "Use ONLY the information provided in the context below to answer the patient question. "
        "Be warm, helpful, and professional. Keep answers clear and concise.\n"
        "If the information is not in the context, say: "
        "I'm sorry, I don't have that information. Please contact us directly at "
        "+977 9866578969 or email neupanedental@gmail.com and our team will be happy to help.\n\n"
        "Context:\n{context}\n\n"
        "Patient Question: {question}"
    ),
)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

print("RAG pipeline ready. Starting server...")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"error": "Empty message"}), 400
    answer = rag_chain.invoke(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
