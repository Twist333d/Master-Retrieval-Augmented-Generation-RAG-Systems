from pprint import pprint
from uuid import uuid4
from helper_utils import word_wrap
import os
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import numpy as np
import chromadb
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import chromadb.utils.embedding_functions as embedding_functions
from collections import Counter

load_dotenv()

# initialize OpenAI client
openai_key = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load PDF
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter for empty strings
pdf_texts = [text for text in pdf_texts if text]

# Split into characters
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

# Split into chunks
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# Initialize Chroma
chroma_client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )
chroma_collection = chroma_client.get_or_create_collection(
    "microsoft-collection", embedding_function=openai_ef
)

# Create a collection
ids = [str(uuid4()) for i in range(len(token_split_texts))] # create unique uuids
chroma_collection.add(
    ids=ids,
    documents=token_split_texts
)

# Get original documents
QUERY = "What were the most important factors that contributed to increases in revenue?"

response = chroma_collection.query(
    query_texts=[QUERY],
    n_results=5,
    include=["documents", "embeddings", 'distances']
)

#pprint(results)
original_results = response
retrieved_documents = response['documents'][0]
retrieved_distances = response['distances'][0]


#print("ORIGINAL RETRIEVED DOCUMENTS:")
#for doc, distance in zip(retrieved_documents, retrieved_distances):
#    print(f"Distance: {distance:.2f}")
#    print(word_wrap(doc))
#    print("\n")

# setup cross encoder
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = QUERY
pairs = [[query, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs) # give us scores for each pair

#print("Scores:")
#for score in scores:
#    print(score)
#    print("\n")

#print("New order:")
#for o in np.argsort(scores)[::-1]:
#    print(f"Document No: {o + 1}")
#    print(word_wrap(retrieved_documents[o]))

original_query = QUERY
# Setup multi-query expansion
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)
def generate_multi_query(query, model='gpt-4o-mini'):
    prompt = """
    You are a helpful expert financial research assistant.
    Your users are inquiring about an annual report. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering."""

    messages =[
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': query}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split('\n')
    return content

generated_queries = generate_multi_query(original_query)
joint_query = [query] + generated_queries

#print("Printing all queries:")
#for q in joint_query:
#    print(q)
#    print("\n")

# Get additional documents
expanded_results = chroma_collection.query(
    query_texts=joint_query,
    n_results=5,
    include=["documents", 'distances']
)

print("EXPANDED RETRIEVED DOCUMENTS:")
pprint(expanded_results)
expanded_documents = expanded_results['documents'][0]
expanded_doc_distances = expanded_results['distances'][0]

print("Expanded retrieved documents:")
for doc, distance in zip(expanded_documents, expanded_doc_distances):
    print(f"Distance: {distance:.2f}")
    print(word_wrap(doc))
    print("\n")

# de-duplicate
unique_documents = {}
for doc, distance in zip(expanded_documents, expanded_doc_distances):
    if doc not in unique_documents or distance < unique_documents[doc]:
        unique_documents[doc] = distance

# Re-rank the document
pairs = []
for doc in unique_documents:
    pairs.append([original_query, doc]) # calculate scores between the original query and the docs

scores = cross_encoder.predict(pairs)

top_indices = np.argsort(scores)[::-1][:5] # select only 5
top_documents = [list(unique_documents.keys())[i] for i in top_indices]
top_distances = [unique_documents[doc] for doc in top_documents]


top_documents_with_distances = list(zip(top_documents, top_distances))
top_5_distances = [distance for _, distance in top_documents_with_distances]

print("Top 5 documents with distances:")
for doc, distance in top_documents_with_distances:
    print(f"Distance: {distance:.4f}")
    print(word_wrap(doc))
    print("\n")

# Concat the docs in a single context
context = "\n\n".join(top_documents)
pprint(context)



def print_summary_stats(original_results):
    print("\n==== Summary Statistics ====\n")

    # 1. Average distance for original and re-ranked results
    original_avg_distance = np.mean(original_results['distances'][0])
    mean_distance_top_5 = np.mean(top_5_distances)

    print(f"Original query average distance: {original_avg_distance:.4f}")
    print(f"Re-ranked average distance: {mean_distance_top_5:.4f}")
    print(f"Distance improvement: {original_avg_distance - mean_distance_top_5:.4f}")

    # 5. Overall improvement percentage
    improvement_percentage = ((original_avg_distance - mean_distance_top_5) / original_avg_distance) * 100
    print(f"\nOverall distance improvement percentage: {improvement_percentage:.2f}%")

# Add this line at the end of your script to call the function
#print_summary_stats(original_results)


# checking the results with Cohere
import cohere
co = cohere.Client(os.getenv("COHERE_API_KEY"))

original_query = QUERY
docs = expanded_results['documents'][0]
top_n = 5 # number of results to return
cohere_results  = co.rerank(model="rerank-english-v3.0", query=query, documents=docs, top_n=5,
return_documents=True)

pprint(cohere_results )

# Process Cohere results
cohere_reranked_docs = [result.document for result in cohere_results]
cohere_scores = [result.relevance_score for result in cohere_results]

# Find indices and distances in the original expanded documents
original_indices = []
original_distances = []

for doc in cohere_reranked_docs:
    for i, doc_list in enumerate(expanded_documents):
        if doc in doc_list:
            original_indices.append(i)
            original_distances.append(expanded_results["distances"][i][doc_list.index(doc)])
            break

# Calculate statistics for Cohere re-ranked results
def print_relevance_stats(distances, scores, query_type):
    print(f"\n{query_type} Query Statistics:")
    print(f"Mean distance: {np.mean(distances):.4f}")
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Min distance: {np.min(distances):.4f}")
    print(f"Max distance: {np.max(distances):.4f}")
    print(f"Standard deviation of distances: {np.std(distances):.4f}")

# Print statistics for original and Cohere re-ranked results
print_relevance_stats(original_distances, [-1] * len(original_distances), "Original")  # Using -1 as placeholder for original scores
print_relevance_stats(original_distances, cohere_scores, "Cohere Re-ranked")

# Compare improvement
mean_distance_improvement = np.mean(original_distances) - np.mean(expanded_results["distances"][0][:5])
print(f"\nMean distance improvement: {mean_distance_improvement:.4f}")

# Print re-ranked results
print("\nCohere Re-ranked Results:")
for i, (doc, score, distance) in enumerate(zip(cohere_reranked_docs, cohere_scores, original_distances)):
    print(f"Rank {i+1}:")
    print(f"Document: {doc[:100]}...")  # Print first 100 characters
    print(f"Cohere Score: {score:.4f}")
    print(f"Original Distance: {distance:.4f}")
    print(f"Original Query Index: {original_indices[i]}")
    print()

# Additional analysis
print("\nAdditional Analysis:")
print(f"Number of documents changed in top 5: {len(set(original_indices[:5]) - set(range(5)))}")
print(f"Average Cohere score: {np.mean(cohere_scores):.4f}")