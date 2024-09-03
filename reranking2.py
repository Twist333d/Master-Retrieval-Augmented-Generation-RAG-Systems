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

results = chroma_collection.query(
    query_texts=[QUERY],
    n_results=5,
    include=["documents", "embeddings", 'distances']
)

#pprint(results)
retrieved_documents = results['documents'][0]
retrieved_distances = results['distances'][0]


print("ORIGINAL RETRIEVED DOCUMENTS:")
for doc, distance in zip(retrieved_documents, retrieved_distances):
    print(f"Distance: {distance:.2f}")
    print(word_wrap(doc))
    print("\n")

# setup cross encoder
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = QUERY
pairs = [[query, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs) # give us scores for each pair

print("Scores:")
for score in scores:
    print(score)
    print("\n")

print("New order:")
for o in np.argsort(scores)[::-1]:
    print(f"Document No: {o + 1}")
    print(word_wrap(retrieved_documents[o]))

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

print("Printing all queries:")
for q in joint_query:
    print(q)
    print("\n")

# Get additional documents

# Setup re-ranker

# Re-rank the document