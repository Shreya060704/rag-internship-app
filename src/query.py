import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import cohere


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-internship-index"
index = pc.Index(index_name)


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

co = cohere.Client(COHERE_API_KEY)

def query_with_reranker(query, top_k=5):
    
    query_vector = embedding_model.encode(query).tolist()

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    docs = [match["metadata"]["text"] for match in results["matches"]]


    rerank_results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )

    print(f"\n🔎 Query: {query}")
    print("📌 Reranked results (most relevant first):")
    for r in rerank_results.results:
        doc = docs[r.index]
        print(f"- (relevance: {r.relevance_score:.4f}) {doc}")

if __name__ == "__main__":
    while True:
        user_query = input("\nAsk something (or type 'exit'): ")
        if user_query.lower() in ["exit", "quit"]:
            break
        query_with_reranker(user_query)
