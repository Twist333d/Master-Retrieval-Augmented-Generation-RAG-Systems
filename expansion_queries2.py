import umap

from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI, embeddings
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()


QUERY = "How is the company's cash flow doing?"

reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# split texts into chunks
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0 # split
    # recursively in the order of priority as above
)
# This gives us text split by characters
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))


token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
# this gives us chunks
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)


import chromadb
from chromadb.utils import embedding_functions
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small"
            )
openai_ef = embedding_functions.DefaultEmbeddingFunction()

# Instantiate the client
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    'microsoft-collection',
    embedding_function=openai_ef
)

# Prepare embeddings
ids = [str(i) for i in range (len(token_split_texts))] # create a list of ids
documents = token_split_texts
chroma_collection.add(ids=ids, documents=documents)
count = chroma_collection.count()
#print(f"Total documents: {count}")


results = chroma_collection.query(query_texts=[QUERY], n_results=5)
retrieved_documents = results['documents'][0] # it's a first element in the list

#print("ORIGINAL RETRIEVAL")
#for document in retrieved_documents:
#    print(word_wrap(document))
#    print("\n ======== \n")

def generate_multi_query(query, model='gpt-3.5-turbo'):
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

original_query = QUERY
aug_queries = generate_multi_query(original_query)

# show the augmented queries
#for query in aug_queries:
#    print(query)
#    print("\n ======== \n")

# 2. Concatenate original query with augmented queries
joint_query = [
    original_query
] + aug_queries

print("======> \n\n", joint_query)

results = chroma_collection.query(
    query_texts = joint_query,
    n_results = 5,
    include=['distances', 'documents', 'embeddings'],
)
retrieved_documents = results["documents"]

#print('Taking a look at the results/')
#pprint(results)

# deduplicate documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

# output the results documents
for i, documents in enumerate(retrieved_documents):
    print(f"Query: {joint_query[i]}")
    print("")
    print("Results:")
    for doc in documents:
        print(word_wrap(doc))
        print("")
    print("-" * 100)


embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# 4. We can also visualize the results in the embedding space
original_query_embedding = openai_ef([original_query])
augmented_query_embeddings = openai_ef(joint_query)


project_original_query = project_embeddings(original_query_embedding, umap_transform)
project_augmented_queries = project_embeddings(
    augmented_query_embeddings, umap_transform
)

retrieved_embeddings = results["embeddings"]
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]

projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)

import matplotlib.pyplot as plt


# Plot the projected query and retrieved documents in the embedding space
plt.figure()
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    project_augmented_queries[:, 0],
    project_augmented_queries[:, 1],
    s=150,
    marker="X",
    color="orange",
)
plt.scatter(
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    project_original_query[:, 0],
    project_original_query[:, 1],
    s=150,
    marker="X",
    color="r",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot

import numpy as np
from collections import Counter


def print_summary_stats(original_results, augmented_results, original_query, augmented_queries):
    print("\n==== Summary Statistics ====\n")

    # 1. Average distance for original and augmented queries
    original_avg_distance = np.mean(original_results['distances'][0])
    augmented_avg_distances = [np.mean(distances) for distances in augmented_results['distances']]
    overall_augmented_avg_distance = np.mean(augmented_avg_distances)

    print(f"Original query average distance: {original_avg_distance:.4f}")
    print(f"Augmented queries average distance: {overall_augmented_avg_distance:.4f}")
    print(f"Distance improvement: {original_avg_distance - overall_augmented_avg_distance:.4f}")

    # 2. Unique documents retrieved
    original_docs = set(original_results['documents'][0])
    augmented_docs = set(doc for sublist in augmented_results['documents'] for doc in sublist)

    print(f"\nUnique documents from original query: {len(original_docs)}")
    print(f"Unique documents from augmented queries: {len(augmented_docs)}")
    print(f"New unique documents found: {len(augmented_docs - original_docs)}")

    # 3. Query performance
    best_query_index = np.argmin(augmented_avg_distances)
    best_query = augmented_queries[best_query_index]
    print(f"\nBest performing query: '{best_query}'")
    print(f"Best query average distance: {augmented_avg_distances[best_query_index]:.4f}")

    # 4. Document frequency
    all_docs = [doc for sublist in augmented_results['documents'] for doc in sublist]
    doc_frequency = Counter(all_docs)
    most_common_doc = doc_frequency.most_common(1)[0]
    print(f"\nMost frequently retrieved document (across all queries): {most_common_doc[1]} times")

    # 5. Overall improvement percentage
    improvement_percentage = ((
                                          original_avg_distance - overall_augmented_avg_distance) / original_avg_distance) * 100
    print(f"\nOverall improvement percentage: {improvement_percentage:.2f}%")


# Usage:
original_results = chroma_collection.query(query_texts=[original_query], n_results=5,
                                           include=['distances', 'documents'])
augmented_results = results  # This is already the result of your augmented queries

print_summary_stats(original_results, augmented_results, original_query, aug_queries)
