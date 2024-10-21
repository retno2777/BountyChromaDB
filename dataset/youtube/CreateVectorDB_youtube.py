import json
import chromadb
from sentence_transformers import SentenceTransformer
import os
import uuid
from tqdm import tqdm

def process_and_save_json_to_chroma(json_file, collection_name, persist_directory=None, batch_size=100, distance_function="cosine"):
    """
    Reads a JSON file, processes the data, creates embeddings, and saves to ChromaDB with a specified distance function.

    :param json_file: Path to the input JSON file.
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
        # Create the collection with the specified distance function in metadata
        collection_metadata = {"hnsw:space": distance_function}
        collection = chroma_client.create_collection(name=collection_name, metadata=collection_metadata)
        print(f"Collection '{collection_name}' created with distance function '{distance_function}'.")

    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load model SentenceTransformer for creating embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Process data in batches
    for start_idx in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch_data = data[start_idx:start_idx + batch_size]
        
        combined_texts = []
        metadatas = []
        ids = []

        # For each item in the batch, combine title and description as text
        for item in batch_data:
            combined_text = f"Title: {item['title']}. Description: {item['description']}."
            combined_texts.append(combined_text)

            # Add relevant metadata, including title
            metadatas.append({
                "videoId": item['videoId'],
                "publishedAt": item['publishedAt'],
                "title": item['title']  # Add title to metadata
            })
            # Create unique ID for each entry
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

json_file = 'youtube_video_details.json'  # Path to your JSON and name file JSON 
collection_name = "videos_collection"
distance_function = "cosine"  # Set the desired distance function ('cosine', 'l2', or 'ip')
process_and_save_json_to_chroma(json_file, collection_name, distance_function=distance_function)
