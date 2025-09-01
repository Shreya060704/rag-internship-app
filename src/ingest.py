import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-internship-index"

if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Created new Pinecone index: {index_name}")

index = pc.Index(index_name)


def load_documents(data_folder="data"):
    documents = []
    for file in os.listdir(data_folder):
        path = os.path.join(data_folder, file)

        if file.endswith(".txt") or file.endswith(".md"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append({"text": text, "source": file})

        elif file.endswith(".pdf"):
            try:
                reader = PdfReader(path)
                text = " ".join(
                    [page.extract_text() for page in reader.pages if page.extract_text()]
                )
                documents.append({"text": text, "source": file})
            except Exception as e:
                print(f"⚠️ Skipping {file} (invalid PDF): {e}")

    return documents

docs = load_documents("data")
print(f"Loaded {len(docs)} documents from data/ folder")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks = []
for doc in docs:
    doc_chunks = splitter.create_documents([doc["text"]])
    for chunk in doc_chunks:
        chunk.metadata["source"] = doc["source"]  
        chunks.append(chunk)

print(f"Split into {len(chunks)} chunks")


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

vectors = []
for i, chunk in enumerate(chunks):
    vector = embedding_model.encode(chunk.page_content).tolist()
    vectors.append((
        f"doc-{i}", 
        vector,
        {"text": chunk.page_content, "source": chunk.metadata.get("source", "unknown")}
    ))

index.upsert(vectors)

print(f"Successfully uploaded {len(vectors)} chunks to Pinecone index '{index_name}'")
