import os

import chromadb
import pylab as p
import umap

from helper_utils import project_embeddings, word_wrap, load_chroma
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

QUERY = "What were the most important factors that contributed to increases in revenue?"

# load environment variables
load_dotenv()

# setup OpenAI client
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# import documents
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# filter for empty strings
pdf_texts = [text for text in pdf_texts if text]

# break into characters
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_text = character_splitter.split_text("\n\n".join(pdf_texts))

# break into chunks
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts=[]
for text in character_split_text:
    token_split_texts += token_splitter.split_text(text)

# create embedding function
from chromadb.utils import embedding_functions
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small"
            )
openai_ef = embedding_functions.DefaultEmbeddingFunction()

# create chroma db
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    'microsoft-collection', embedding_function=openai_ef
)

# create collections
ids = [str(i) for i in range(len(token_split_texts))] # create a list of ids
chroma_collection.add(
    ids=ids, # required to add
    documents=token_split_texts
)

count = chroma_collection.count()
print(f"Number of documents: {count}")

query = QUERY

original_results = chroma_collection.query(
    query_texts=query, n_results=5, include=["documents", "embeddings", "distances"]
)

retrieved_documents = original_results['documents'][0]

#for doc in retrieved_documents:
#    print(word_wrap(doc))
#    print("====")

# setup cross-encoder to rank documents
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [[query, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs)

#print("Scores:")
#for score in scores:
#    print(score)

#print("New Ordering:")
#for o in np.argsort(scores)[::-1]:
#    print(o + 1)


def generate_multi_query(query, model="gpt-4o-mini"):

    prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
                """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content


original_query = (
    QUERY
)
aug_queries = generate_multi_query(original_query)

# 1. First step show the augmented queries
print("Augmented Queries:")
for query in aug_queries:
    print("\n", query)

# 2. concatenate the original query with the augmented queries
joint_query = [
    original_query
] + aug_queries  # original query is in a list because chroma can actually handle multiple queries, so we add it in a list

# re run the search

augmented_results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings", 'distances']
)
expanded_retrieved_documents = augmented_results["documents"]

# Deduplicate the retrieved documents
unique_documents = set()
for documents in expanded_retrieved_documents:
    for document in documents:
        unique_documents.add(document)

unique_documents = list(unique_documents)

pairs = []
for doc in unique_documents:
    pairs.append([original_query, doc])

scores = cross_encoder.predict(pairs)

print("Scores:")
for score in scores:
    print(score)

print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o)
# ====
top_indices = np.argsort(scores)[::-1][:5]
top_documents = [unique_documents[i] for i in top_indices]

# Concatenate the top documents into a single context
context = "\n\n".join(top_documents)
print("Context:")
print(context)


import numpy as np

def print_relevance_stats(distances, query_type):
    print(f"\n{query_type} Query Statistics:")
    print(f"Mean distance: {np.mean(distances):.4f}")
    print(f"Min distance: {np.min(distances):.4f}")
    print(f"Max distance: {np.max(distances):.4f}")
    print(f"Standard deviation: {np.std(distances):.4f}")

# For the original query
original_results = chroma_collection.query(
    query_texts=[original_query],
    n_results=5,
    include=["distances"]
)
original_distances = original_results['distances'][0]

# For the re-ranked results
top_indices = np.argsort(scores)[::-1][:5]  # Get indices of top 5 re-ranked documents

# Find the corresponding distances for the top 5 re-ranked documents
re_ranked_top_5_distances = []
for idx in top_indices:
    doc = unique_documents[idx]
    # Find the original distance for this document
    original_idx = augmented_results["documents"][0].index(doc)
    distance = augmented_results["distances"][0][original_idx]
    re_ranked_top_5_distances.append(distance)

# Print statistics
print_relevance_stats(original_distances, "Original")
print_relevance_stats(re_ranked_top_5_distances, "Re-ranked")

# Compare improvement
mean_improvement = np.mean(original_distances) - np.mean(re_ranked_top_5_distances)
print(f"\nMean distance improvement: {mean_improvement:.4f}")

# Percentage of improved results
improved_count = sum(r < o for r, o in zip(re_ranked_top_5_distances, original_distances))
print(f"Percentage of improved results: {improved_count / len(re_ranked_top_5_distances) * 100:.2f}%")

# TODO
# Calculate improvement in distance scores
# Show how will this work with any other type of PDF (latest example)
# Post about it
# Update my plan for RAG over docs
# Integrate multi-query expansion + re-ranking


