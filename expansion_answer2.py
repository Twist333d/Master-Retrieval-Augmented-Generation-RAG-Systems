import umap

from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI, embeddings
from dotenv import load_dotenv
from pprint import pprint


QUERY = "How did Gaming segment perform this year in comparison to the previous?"

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)


reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# filter empty strings
#pdf_texts = [text for text in pdf_texts if text]
#print(
#    word_wrap(
#        pdf_texts[0],
#        width=100,
#    )
#)

# Split text into chunks
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0 # split
    # recursively in the order of priority as above
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))
#print(word_wrap(character_split_texts[10]))
#print(f"\nTotal chunks: {len(character_split_texts)}")

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

#print(word_wrap(token_split_texts[10]))
#print(f"\nTotal chunks: {len(token_split_texts)}")

import chromadb
from chromadb.utils import embedding_functions
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small"
            )
openai_ef = embedding_functions.DefaultEmbeddingFunction()
#embedding = openai_ef(token_split_texts[10])
#print(embedding)

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
retrieved_distances = results['distances'][0] # it's a first element in the list

print("ORIGINAL RETRIEVAL")
for document in retrieved_documents:
    print(word_wrap(document))
    print("\n ======== \n")


def augment_query_generated(query, model="gpt-3.5-turbo"):
    prompt = """
    You are a helpful financial research assistant. Provide an example answer to the given 
    question, that might be found in a document like an annual report."""

    messages = [
        {
            'role' : 'system',
            'content' : prompt,
        },
        {'role' : 'user', 'content' : query},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages, # pass in the list with dictionary of answers
    )
    content = response.choices[0].message.content
    return content

original_query = QUERY
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"
print(word_wrap(joint_query))
print("========")

results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings", "distances"]
)

#print("Studying the results:")
#pprint(results)

retrieved_documents = results['documents'][0]

print("WITH JOINT QUERY: ")
for doc in retrieved_documents:
    print(word_wrap(doc))
    print("\n ======== \n")

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)


retrieved_embeddings = results["embeddings"][0]
original_query_embedding = openai_ef([original_query])
augmented_query_embedding = openai_ef([joint_query])

projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

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
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot

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

# For the joint query
joint_results = chroma_collection.query(
    query_texts=[joint_query],
    n_results=5,
    include=["distances"]
)
joint_distances = joint_results['distances'][0]

# Print statistics
print_relevance_stats(original_distances, "Original")
print_relevance_stats(joint_distances, "Joint")

# Compare improvement
mean_improvement = np.mean(original_distances) - np.mean(joint_distances)
print(f"\nMean distance improvement: {mean_improvement:.4f}")

# Percentage of improved results
improved_count = sum(j < o for j, o in zip(joint_distances, original_distances))
print(f"Percentage of improved results: {improved_count / len(joint_distances) * 100:.2f}%")



