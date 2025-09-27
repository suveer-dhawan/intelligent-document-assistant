"""
This script creates a vector store from a dataset of Netflix movie reviews using Ollama embeddings and Chroma as the vector store.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

def create_rich_content(row):
    """Create rich, searchable content from Netflix data"""
    # Handle missing values
    title = str(row["title"]) if pd.notna(row["title"]) else "Unknown Title"
    content_type = str(row["type"]) if pd.notna(row["type"]) else "Unknown Type"
    description = str(row["description"]) if pd.notna(row["description"]) else "No description available"
    release_year = str(int(row["release_year"])) if pd.notna(row["release_year"]) else "Unknown Year"
    genres = str(row["genres"]) if pd.notna(row["genres"]) else "Unknown Genre"
    age_cert = str(row["age_certification"]) if pd.notna(row["age_certification"]) else "Not Rated"
    runtime = str(int(row["runtime"])) if pd.notna(row["runtime"]) else "Unknown Runtime"
    countries = str(row["production_countries"]) if pd.notna(row["production_countries"]) else "Unknown Countries"
    imdb_score = str(row["imdb_score"]) if pd.notna(row["imdb_score"]) else "No IMDB Score"
    seasons = str(int(row["seasons"])) if pd.notna(row["seasons"]) and row["seasons"] > 0 else ""
    
    # Create rich, searchable content
    content_parts = [
        f"Title: {title}",
        f"Type: {content_type}",
        f"Released: {release_year}",
        f"Genres: {genres}",
        f"Age Rating: {age_cert}",
        f"Runtime: {runtime} minutes" if runtime != "Unknown Runtime" else "",
        f"Countries: {countries}",
        f"IMDB Score: {imdb_score}" if imdb_score != "No IMDB Score" else "",
        f"Seasons: {seasons}" if seasons else "",
        f"Description: {description}"
    ]
    
    # Join non-empty parts
    return " | ".join([part for part in content_parts if part])

def load_netflix_data(csv_path="titles.csv"):
    """Load and process Netflix dataset"""
    df = pd.read_csv(csv_path)
    return df


# Initialize the Ollama Embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Check if the vector store already exists
db_location = "./netflix_chroma_db"
add_documents = not os.path.exists(db_location)

# Load Netflix data
df = load_netflix_data()

# Create documents and their IDs if the vector store is being created for the first time
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():        
        
        content = create_rich_content(row)
        
        # Create comprehensive metadata
        metadata = {
            "title": str(row["title"]) if pd.notna(row["title"]) else "Unknown",
            "type": str(row["type"]) if pd.notna(row["type"]) else "Unknown",
            "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else 0,
            "genres": str(row["genres"]) if pd.notna(row["genres"]) else "Unknown",
            "age_certification": str(row["age_certification"]) if pd.notna(row["age_certification"]) else "Not Rated",
            "runtime": int(row["runtime"]) if pd.notna(row["runtime"]) else 0,
            "production_countries": str(row["production_countries"]) if pd.notna(row["production_countries"]) else "Unknown",
            "imdb_score": float(row["imdb_score"]) if pd.notna(row["imdb_score"]) else 0.0,
            "imdb_votes": int(row["imdb_votes"]) if pd.notna(row["imdb_votes"]) else 0,
            "tmdb_score": float(row["tmdb_score"]) if pd.notna(row["tmdb_score"]) else 0.0,
            "seasons": int(row["seasons"]) if pd.notna(row["seasons"]) and row["seasons"] > 0 else 0
        }
        
        doc = Document(
            page_content=content,
            metadata=metadata,
            id=str(i)
        )
        
        documents.append(doc)
        ids.append(str(i))

vector_store = Chroma(
    collection_name="netflix_titles",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents to the vector store if it is newly created
if add_documents:

    print("Adding documents to vector store...")
    
    # Process in batches to avoid ChromaDB batch size limit
    batch_size = 1000  # Safe batch size
    total_docs = len(documents)
    
    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        batch_docs = documents[i:end_idx]
        batch_ids = ids[i:end_idx]
        
        print(f"   Adding batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} ({end_idx}/{total_docs} documents)")
        vector_store.add_documents(batch_docs, ids=batch_ids)
    
    print("Vector store created and populated successfully!")
    vector_store.add_documents(documents, ids=ids)

# Create a retriever from the vector store
retriever= vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)