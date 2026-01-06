import os
import glob
import numpy as np
from dotenv import load_dotenv
from agent_config import *
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sklearn.manifold import TSNE
import plotly.graph_objects as go


# agent, MODEL = initialize_agent(llm_type="local")

folders = glob.glob("kb/*")
db_name = "vector_db"


# Go through folders and mark docs with doc type
documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)
# for doc in documents:
#     print(doc.metadata["doc_type"])

# Split doc texts into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
# print(f"Divided into {len(chunks)} chunks")
# print(f"First chunk:\n\n{chunks[1]}")


# Create Chroma vectorstore(with clearing previous data), get vector and find how many dimensions it has
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")
collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

# Prework on data format
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
colors = [['orange', 'green', 'blue'][['ai_demos', 'optimize_codes', 'python_tests'].index(t)] for t in doc_types]

# Reduce the dimensionality of the vectors to 2D using t-SNE
# (t-distributed stochastic neighbor embedding)

tsne = TSNE(n_components=2, random_state=42, perplexity=3)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x',yaxis_title='y'),
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)


# # 3D version
# # Create the 3D scatter plot
# fig = go.Figure(data=[go.Scatter3d(
#     x=reduced_vectors[:, 0],
#     y=reduced_vectors[:, 1],
#     z=reduced_vectors[:, 2],
#     mode='markers',
#     marker=dict(size=5, color=colors, opacity=0.8),
#     text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
#     hoverinfo='text'
# )])
#
# fig.update_layout(
#     title='3D Chroma Vector Store Visualization',
#     scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
#     width=900,
#     height=700,
#     margin=dict(r=20, b=10, l=10, t=40)
# )

fig.show()
