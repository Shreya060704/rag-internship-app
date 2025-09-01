import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import cohere
from transformers import pipeline

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-internship-index"
index = pc.Index(index_name)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
co = cohere.Client(COHERE_API_KEY)


answer_generator = pipeline("summarization", model="facebook/bart-large-cnn")

st.set_page_config(page_title="RAG Internship App", page_icon="📘", layout="wide")

st.title("📘 RAG Internship Project")
st.markdown("### Retrieval-Augmented Generation with Pinecone, Cohere & Hugging Face 🚀")

st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Top K Results", 1, 10, 3)
relevance_threshold = st.sidebar.slider("Relevance Threshold", 0.0, 1.0, 0.3)


query = st.text_input("🔎 Ask a question:")

if query:
    
    query_vector = embedding_model.encode(query).tolist()


    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    docs = [match["metadata"]["text"] for match in results["matches"]]
    sources = [match["metadata"].get("source", "unknown") for match in results["matches"]]

    
    rerank_results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )

    
    if all(r.relevance_score < relevance_threshold for r in rerank_results.results):
        st.warning("⚠️ Sorry, I couldn’t find a confident answer in the documents.")
    else:
       
        st.subheader("📌 Retrieved & Reranked Results")
        for idx, r in enumerate(rerank_results.results, start=1):
            doc = docs[r.index]
            source = sources[r.index]
            rerank_score = r.relevance_score  

            st.markdown(
                f"""
                <div style="padding:15px; margin-bottom:15px; border-radius:12px; 
                            background-color:#1e1e2f; border:1px solid #444;">
                <b style="color:#ff4c4c;">[{idx}] Relevance:</b> {rerank_score:.4f}<br>
                <b style="color:#00c3ff;">Source:</b> {source}<br>
                <b style="color:#fff;">Text:</b> {doc}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        
        if st.button("💡 Generate Answer with Hugging Face"):
        
            context = " ".join([docs[r.index] for r in rerank_results.results])

            response = answer_generator(
                f"Question: {query}\nContext: {context}",
                max_length=120,
                min_length=40,
                do_sample=False
            )
            answer = response[0]["summary_text"]

          
            citations = ", ".join([f"[{i+1}]" for i in range(len(rerank_results.results))])
            final_answer = f"{answer} {citations}"

            st.subheader("💬 Generated Answer")
            st.markdown(
                f"""
                <div style="padding:20px; border-radius:15px; background-color:#2a2a40; 
                            color:#e6e6e6; font-size:16px; line-height:1.6;">
                {final_answer}
                </div>
                """,
                unsafe_allow_html=True
            )
