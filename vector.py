from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load the dataset
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Initialize the Ollama Embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Check if the vector store already exists
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# Create documents and their IDs if the vector store is being created for the first time
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():        
        
        doc = Document(
            page_content=row["Title"] + " " + row["Review"], 
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id = str(i)
        ) 
        
        ids.append(str(i))
        documents.append(doc)

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents to the vector store if it is newly created
if add_documents:
    vector_store.add_documents(documents, ids=ids)

# Create a retriever from the vector store
retriever= vector_store.as_retriever(
    search_kwargs={"k": 5}
)