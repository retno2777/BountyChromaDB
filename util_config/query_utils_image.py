from util_config.config_model import chroma_client, clip_model, tokenize_image, reduce_embedding_dimension
import torch

def query_all_collections_with_image(image, top_k=1, distance_type='cosine'):
    """
    Perform a query on three collections (articles, YouTube videos, and images) using the given image as query input
    based on the selected distance type from the frontend and combine the results from the three collections.
    
    :param image: The image input to search in the collections.
    :param top_k: The number of top results to retrieve based on similarity.
    :param distance_type: The type of distance to use ('cosine', 'l2', 'ip').
    :return: A list of search results from the three collections.
    """
    search_results = []

    # Determine the collection name based on the selected distance type
    if distance_type == 'cosine':
        article_collection_name = "articles_collection"
        youtube_collection_name = "videos_collection"
        image_collection_name = "image_collection"
    else:
        article_collection_name = f"articles_collection_{distance_type}"
        youtube_collection_name = f"videos_collection_{distance_type}"
        image_collection_name = f"image_collection_{distance_type}"

    # Preprocess the image for CLIP model
    image_input = tokenize_image(image)  # This should tokenize or preprocess the image for the CLIP model
    with torch.no_grad():
        image_embeds = clip_model.encode_image(image_input)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    # Convert embeddings to a list to query in ChromaDB
    query_embedding_images = image_embeds.cpu().numpy().flatten().tolist()
    
    #Reduce dimention for article and youtube
    query_embedding_images_reduce = reduce_embedding_dimension(query_embedding_images)

    # Query the article collection using image embedding
    article_collection = chroma_client.get_collection(article_collection_name)
    article_results = article_collection.query(query_embeddings=[query_embedding_images_reduce], n_results=top_k)

    if article_results['documents']:
        similarity = 1 - article_results['distances'][0][0]  # Calculate similarity for all distance types
        metadata = article_results['metadatas'][0][0]
        search_results.append({
            'type': 'article',
            'title': metadata.get('title', 'Unknown'),
            'summary': metadata.get('summary', 'No description available'),
            'url': metadata.get('url', 'Unknown'),
            'similarity': round(similarity, 4)
        })

    # Query the YouTube video collection using image embedding
    youtube_collection = chroma_client.get_collection(youtube_collection_name)
    youtube_results = youtube_collection.query(query_embeddings=[query_embedding_images_reduce], n_results=top_k)

    if youtube_results['documents']:
        metadata = youtube_results['metadatas'][0][0]  # Get the first metadata
        similarity = 1 - youtube_results['distances'][0][0]  # Calculate similarity
        search_results.append({
            'type': 'youtube',
            'video_id': metadata.get('videoId', 'Unknown'),
            'title': metadata.get('title', 'Unknown'),
            'similarity': round(similarity, 4)
        })

    # Query the image collection using image embedding
    image_collection = chroma_client.get_collection(image_collection_name)
    image_results = image_collection.query(query_embeddings=[query_embedding_images], n_results=top_k)

    if image_results['documents']:
        similarity = 1 - image_results['distances'][0][0]  # Calculate similarity for all distance types
        metadata = image_results['metadatas'][0][0]
        search_results.append({
            'type': 'image',
            'image_path': metadata.get('file_name', 'Unknown'),
            'similarity': round(similarity, 4)
        })

    # Sort the results by similarity in descending order
    search_results.sort(key=lambda x: x['similarity'], reverse=True)

    return search_results


