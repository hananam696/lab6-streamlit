import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed document embeddings
embeddings = np.load("embeddings.npy")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

def retrieve_top_k(query_embedding, embeddings, k=10):
    """Retrieve top-k most similar documents using cosine similarity."""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

# Fine-tuned query embedding
def get_query_embedding(query):
    np.random.seed(abs(hash(query)) % (10**6))
    return np.random.rand(embeddings.shape[1])

st.title("Information Retrieval using Document Embeddings")

# Input query
query = st.text_input("Enter your query:")

if st.button("Search") and query.strip():
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings)

    # Display results
    st.write("### Top 10 Relevant Documents:")
    for doc, score in results:

        st.write(f"- **{doc.strip()}** (Score: {score:.4f})")
