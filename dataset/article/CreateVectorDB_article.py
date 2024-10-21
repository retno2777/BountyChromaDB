import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os
import uuid
from tqdm import tqdm

def process_and_save_to_chroma(csv_file, collection_name, persist_directory=None, batch_size=100, distance_function="cosine"):
    """
    Reads a CSV file, processes the data, creates embeddings, and saves to ChromaDB with a specified distance function.

    :param csv_file: Path to the input CSV file.
    :param collection_name: Name of the collection in ChromaDB.
    :param persist_directory: Path to the ChromaDB persistence directory. If None, defaults to 'chroma_db' in the current directory.
    :param batch_size: Number of rows to process in a single batch.
    :param distance_function: The distance function to use in ChromaDB ('cosine', 'l2', 'ip').
    """
    if persist_directory is None:
        persist_directory = os.path.join('..', 'chroma_db')

    # Create persistence directory if it does not exist
    os.makedirs(persist_directory, exist_ok=True)
    print(f"Using directory: {persist_directory}")
    
    # Initialize PersistentClient ChromaDB
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    
    # Check if collection already exists, create a new one with the specified distance function otherwise
    try:
        collection = chroma_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' loaded successfully.")
    except chromadb.errors.InvalidCollectionException:
        # Create a new collection with the specified distance function
        collection_metadata = {"hnsw:space": distance_function}
        collection = chroma_client.create_collection(name=collection_name, metadata=collection_metadata)
        print(f"Collection '{collection_name}' created with distance function '{distance_function}'.")

    # Read CSV and replace empty columns with "N/A"
    df = pd.read_csv(csv_file)
    df.fillna("N/A", inplace=True)
    df = df.astype(str)

    # Load model SentenceTransformer for creating embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Process data in batches
    for start_idx in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[start_idx:start_idx+batch_size]
        
        combined_texts = []
        metadatas = []
        ids = []

        # Combine title and summary as text for each row in the batch
        for _, row in batch_df.iterrows():
            combined_text = f"Title: {row['title']}. Summary: {row['summary']}."
            if row['title'] == "N/A" and row['summary'] == "N/A":
                combined_text = "No relevant information available"

            combined_texts.append(combined_text)
            
            # Add relevant metadata
            metadatas.append({
                "title": row['title'],
                "summary": row['summary'],
                "published": row['published'],
                "url": row['url'],
                "images": row['images']
            })
            # Generate a unique ID for each entry
            ids.append(str(uuid.uuid4()))

        # Create embeddings using the model
        embeddings = model.encode(combined_texts).tolist()

        # Save embeddings, documents, metadata, and IDs to ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=combined_texts,
            metadatas=metadatas,
            ids=ids
        )

    print(f"Dataset processed and saved to ChromaDB successfully with distance function '{distance_function}'.")

# Contoh penggunaan
csv_file = 'newsArticleDataset_filtered.csv'
collection_name = "articles_collection"
distance_function = "cosine"  # Set the desired distance function ('cosine', 'l2', or 'ip')
process_and_save_to_chroma(csv_file, collection_name, distance_function=distance_function)
