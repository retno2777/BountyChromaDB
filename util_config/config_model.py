import chromadb
from sentence_transformers import SentenceTransformer
import clip
import torch
from PIL import Image
import torch.nn as nn


persist_directory = 'dataset/chroma_db'  
chroma_client = chromadb.PersistentClient(path=persist_directory)


sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def tokenize_text(text_query):
    """
    Tokenizes text using CLIP's tokenizer and moves it to the appropriate device (CPU/GPU).
    
    :param text_query: Teks yang ingin di-tokenize.
    :return: Tokenized text tensor.
    """
    return clip.tokenize([text_query]).to(device)

def tokenize_image(image_path):
    """
    Tokenizes an image using CLIP's preprocessing pipeline and moves it to the appropriate device (CPU/GPU).
    
    :param image_path: Path of the image to be tokenized.
    :return: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")  # Opens the image and ensures RGB format
    image_tensor = preprocess(image).unsqueeze(0).to(device)  # Preprocesses and converts to batch tensor
    return image_tensor


class DimensionalityReducer(nn.Module):
    def __init__(self):
        super(DimensionalityReducer, self).__init__()
        self.linear = nn.Linear(512, 384)
    
    def forward(self, x):
        return self.linear(x)
    
dim_reducer = DimensionalityReducer().to(device)

def reduce_embedding_dimension(embedding):
    """
    Mereduksi embedding CLIP dari 512 dimensi menjadi 384 dimensi menggunakan proyeksi linear.
    
    :param embedding: Embedding CLIP (dengan 512 dimensi).
    :return: Embedding yang direduksi menjadi 384 dimensi.
    """
    with torch.no_grad(): 
        embedding_tensor = torch.tensor(embedding).to(device)
        reduced_embedding = dim_reducer(embedding_tensor)
    
    return reduced_embedding.cpu().numpy().flatten()