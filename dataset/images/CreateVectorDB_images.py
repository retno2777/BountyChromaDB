import os
import uuid
import torch
from PIL import Image
import clip
import chromadb
from tqdm import tqdm

def process_images_and_save_to_chroma(image_folder, collection_name, persist_directory=None, batch_size=32, distance_function="cosine"):
    """
    Reads images from a folder, generates CLIP embeddings, and saves to ChromaDB with specified distance function.

    :param image_folder: Path to the folder containing the images.
    :param collection_name: Name of the collection in ChromaDB.
    :param persist_directory: Path to the ChromaDB persistence directory. If None, will use 'chroma_db' in the current directory.
    :param batch_size: Number of images to process in a single batch.
    :param distance_function: The distance function to use in the collection ("cosine", "l2", "ip").
    """
    if persist_directory is None:
        persist_directory = os.path.join('..', 'chroma_db')

    os.makedirs(persist_directory, exist_ok=True)
    print(f"Using directory: {persist_directory}")
    
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        collection = chroma_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' loaded successfully.")
    except chromadb.errors.InvalidCollectionException:
        # Create the collection with the specified distance function in metadata
        collection_metadata = {"hnsw:space": distance_function}
        collection = chroma_client.create_collection(collection_name, metadata=collection_metadata)
        print(f"Collection '{collection_name}' created with distance function '{distance_function}'.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        batch_file_names = []
        
        for file_name in batch_files:
            try:
                image_path = os.path.join(image_folder, file_name)
                image = Image.open(image_path).convert('RGB')
                batch_images.append(preprocess(image))
                batch_file_names.append(file_name)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        
        if not batch_images:
            continue
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            image_embeds = model.encode_image(batch_tensor)
        
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        embedding_vectors = image_embeds.cpu().numpy().tolist()
        
        unique_ids = [str(uuid.uuid4()) for _ in batch_file_names]
        metadatas = [{"file_name": file_name} for file_name in batch_file_names]
        
        collection.add(
            embeddings=embedding_vectors,
            documents=batch_file_names,
            metadatas=metadatas,
            ids=unique_ids
        )

    print(f"Total {len(image_files)} images have been processed and saved to Chroma DB collection '{collection_name}'.")

# Contoh penggunaan
image_folder = 'image_rgb'
collection_name = 'image_collection_l2'
distance_function = "l2"  # You can also use "l2" or "ip" here.
process_images_and_save_to_chroma(image_folder, collection_name, distance_function=distance_function)
