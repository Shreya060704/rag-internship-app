# rag-internship-app

# Retrieval-Augmented Generation (RAG) Application

This project implements a Retrieval-Augmented Generation (RAG) system using Pinecone, Cohere, Hugging Face, and Streamlit. The system retrieves relevant information from a vector database, reranks the results using a dedicated reranker model, and generates grounded answers with inline citations. The application is deployed on Streamlit Cloud.

# Key Features

1. Embeddings: Query and document embeddings generated with Hugging Face (all-MiniLM-L6-v2, 384-dimensional).

2. Vector Database: Pinecone for efficient storage and retrieval of embeddings.

3. Reranker: Cohere’s rerank-english-v3.0 model to reorder retrieved results for improved relevance.

4. Answer Generation: Summarized answers produced with Hugging Face’s facebook/bart-large-cnn.

5. User Interface: Built with Streamlit, supporting citations and source references.

6. Deployment: Hosted on Streamlit Cloud.


# System Architecture

1. User submits a query.

2. Query is embedded using Hugging Face embeddings.

3. Pinecone retrieves the top-k most relevant document chunks.

4. Cohere reranker reorders the retrieved results.

5. Hugging Face summarizer generates a final grounded answer with inline citations.



# Design Decisions

## Chunking:

1. Chunk size: 500 tokens

2. Chunk overlap: 50 tokens
These values were chosen to balance retrieval granularity with contextual completeness.

3. Embedding Model:
Hugging Face’s all-MiniLM-L6-v2 was selected for its efficiency and compatibility with free-tier vector dimensions.

4. Reranking:
Cohere’s reranker ensures that the most relevant chunks are surfaced first, improving factual grounding.



# Project Structure

rag-internship-app/
│── data/                # Input documents (txt, md, pdf, docx, csv)
│── src/
│   ├── ingest.py        # Ingests and chunks documents, uploads to Pinecone
│   ├── query.py         # Command-line interface for retrieval
│   ├── app.py           # Streamlit application
│── requirements.txt     # Dependencies
│── README.md            # Documentation



# Remarks

- OpenAI embeddings were initially used but replaced with Hugging Face models due to quota limitations.

- Pinecone free-tier (384-dimensional index) was used to remain within assessment constraints.

- The application prioritizes explainability by including citations and document sources with generated answers.


# Evaluation

## Methodology

To evaluate the performance of the RAG application, a small set of five representative queries was selected, each aligned with the contents of the ingested documents. The evaluation process followed these steps:

1. Query Selection: Queries were designed to cover different aspects of the ingested knowledge base (definitions, challenges, applications, and specific components).

2. Ground Truth Definition: For each query, an expected “gold answer” was defined manually based on the source documents.

3. System Response Comparison: The system’s generated answers were compared against the gold answers to determine correctness.

4. Evaluation Criteria: Responses were judged on three main dimensions:

5. Accuracy: Whether the answer captured the correct factual information.

6. Completeness: Whether the answer included all relevant aspects of the expected response.

7. Grounding: Whether the answer was supported by retrieved context (citations and sources).

# Results

1. What is Pinecone?

- Retrieval: Retrieved correct chunk from pinecone_guide.md.

- Generation: Produced a precise and correct definition.

- Result: Correct.

2. What are the challenges of RAG?

- Retrieval: Retrieved relevant section from rag_survey.pdf.

- Generation: Preserved the key points (chunking, noisy retrieval, latency, multimodal integration).

- Result: Correct.

3. What does Cohere reranker do?

- Retrieval: Retrieved description from cohere_reranker.txt.

- Generation: Summarized correctly but concisely.

- Result: Correct.

4. What are the applications of RAG?

- Retrieval: Retrieved relevant sections listing QA, chatbots, enterprise search, research assistants.

- Generation: Preserved and presented the full list.

- Result: Correct.

5. What is AI?

- Retrieval: Retrieved a complete definition from ai_basics.txt (with details about reasoning, learning, problem-solving, etc.).

- Generation: Summarizer only included the first part (“simulation of human intelligence in machines”), omitting details.

- Result: Partially correct (accurate but incomplete).


# Overall Performance

1. Retrieval Accuracy: Strong – relevant chunks were consistently retrieved and reranked correctly.

2. Generation Accuracy: Generally correct, but summarization occasionally produced shorter answers that omitted details.

3. Final Outcome: 4/5 correct answers, 1 partially correct. Approximate accuracy: 80%.

# Key Insight

The evaluation shows that the retrieval and reranking components performed reliably, while some limitations arose from the summarization model producing overly concise outputs. This distinction highlights that the pipeline is effective at surfacing the right information, and answer completeness can be improved by fine-tuning or selecting a different generation model.

# Deployment

The application is publicly hosted on Streamlit Cloud:
[https://rag-internship-app-ex7wk9cedokuxypsy4x7sj.streamlit.app/]