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
QUERY = "How did gaming segment perform this year in comparison to last year?"

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

# Ensure joint_query is a list of strings
joint_query = [str(q) for q in joint_query]

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

#print("Expanded retrieved documents:")
#for doc, distance in zip(expanded_documents, expanded_doc_distances):
#    print(f"Distance: {distance:.2f}")
#    print(word_wrap(doc))
#    print("\n")

# de-duplicate
unique_documents = {}
for doc, distance in zip(expanded_documents, expanded_doc_distances):
    if doc not in unique_documents or distance < unique_documents[doc]:
        unique_documents[doc] = distance


unique_documents_list = list(doc for doc in unique_documents.keys())

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

#print("Top 5 documents with distances:")
#for doc, distance in top_documents_with_distances:
#    print(f"Distance: {distance:.4f}")
#    print(word_wrap(doc))
#    print("\n")

# Concat the docs in a single context
context = "\n\n".join(top_documents)
#pprint(context)

import cohere
co = cohere.Client(os.getenv("COHERE_API_KEY"))

original_query = QUERY
docs = unique_documents_list
top_n = 5 # number of results to return
cohere_results  = co.rerank(model="rerank-english-v3.0", query=query, documents=docs, top_n=top_n,
return_documents=True)

def process_cohere_results(cohere_response, relevance_threshold=0.01, print_results=False):
    """Processes cohere results
    removes results below relevance threshold
    prints the result"""
    processed_results = []

    for result in cohere_response.results:
        relevance_score = result.relevance_score
        if relevance_score >= relevance_threshold:
            doc = result.document.text
            entry = f"Relevance score: {relevance_score:.3f}\n Document: {doc}"
            processed_results.append(entry)

    if print_results:
        print("PRINTING COHERE PROCESSING RESULTS")
        for result in processed_results:
            print(result)

    return processed_results

context = process_cohere_results(cohere_results)
#print(context)

def final_answer_generation(query, context, model="gpt-4o-mini"):

    prompt = (f"You are a knowledgable financial research assistant. Your users are inquiring about an annual report."
              f"You will be given context, extracted by an LLM that will help in answering the "
              f"questions. Each context has a relevance score and the document itself"
              f"If the provided context is not relevant, please inform the user that you can not "
              f"answer the question based on the provided information."
              f"If the provided context is relevant, answer the question based on the contex")

    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': f"Context: \n\n{context}\n\n."
                                    f"Answer the questions based on the provided context."
                                    f"Question: {query}"},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    content = response.choices[0].message.content
    #content = content.split('\n')
    return content

res = final_answer_generation(original_query, context, model="gpt-4o-mini")
print("REPLY:")
print(res)