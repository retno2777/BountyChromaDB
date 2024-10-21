# 😊 ChromaDB App

ChromaDB App is a robust web application designed to efficiently handle both **text** and **image searches** using the **ChromaDB VectorDatabase**. It offers users a seamless, intuitive interface and to input queries and retrieve relevant results from three content categories:

- 📰 **News Articles**
- 🎥 **YouTube Videos**
- 🖼️ **Images**

## 🎯 Purpose
The app allows for easy switching between **text-based** and **image-based** searches, providing flexibility based on the query type. It ensures high-quality, relevant results using advanced **vector similarity search algorithms** such as:

- 📏 **Cosine Similarity**
- 📐 **L2 Distance (Euclidean Distance)**

### 📜 About the ChromaDB Bounty Quest
This project was developed as part of the **ChromaDB Bounty Quest** hosted on the **StackUp** platform. The bounty aims to create an application that leverages the powerful ChromaDB Vector Database for fast and accurate search capabilities. Developers are challenged to showcase their skills by building an app that integrates ChromaDB and demonstrates efficient data processing, vector similarity search, and multi-modal (text and image) search functionalities. The key focus of the quest is to optimize performance, accuracy, and resource management in handling large datasets.

By participating in this bounty, the goal is to push the limits of **vector search** technologies and demonstrate innovative ways to build high-performance applications using **ChromaDB**.

---

## ✨ Features

- 📝 **Text Search**: Input a query to retrieve relevant results from articles, YouTube videos, and images.
- 🖼️ **Image Search**: Upload an image to get relevant results from the same three collections.
- 🔄 **Flexible Search Mode**: Easily toggle between text and image search.
- 🖥️ **User-Friendly Interface**: Results are displayed in a clean, grid-based layout for easy browsing.

## 🛠️ Technologies Used

- 🌐 **Frontend**: HTML, CSS, JavaScript
- 🔧 **Backend**: Flask, ChromaDB
- 💾 **Database**: ChromaDB


## 📚 Creating Article, Image and Youtube Collection for ChromaDB

This section explains how to create and preprocess both the **article collection** and **image collection** from the **Kaggle News Dataset** for use in the **ChromaDB App**. The data will be processed and embedded using appropriate models, making it ready for efficient vector similarity search.

### Steps:

### 1. Dataset Collection
- The dataset is sourced from the **Kaggle News Dataset**, available [here](https://www.kaggle.com/datasets/mdkabinhasan/news-dataset-with-images).
- It contains articles, images, and metadata such as titles, descriptions, and image URLs.
- To collect YouTube video data, first need to obtain an API key. Follow the tutorial [here](https://www.getphyllo.com/post/how-to-get-youtube-api-key) to create and set up your API key.
- Official YouTube API documentation can be found [here](https://developers.google.com/youtube/v3/docs/?apix=true).

### 2. Preprocessing Pipeline for Articles

- **Load the Dataset**:
    Load the dataset using pandas to inspect the data and select relevant columns for articles.

    ```python
    import pandas as pd

    # Load the dataset
    dataset_path = 'path_to_dataset.csv'
    df = pd.read_csv(dataset_path)
    ```
- **Cut Data to 5000 Rows**
    To keep the dataset manageable, limit the number of rows to 5000 for article processing.

    ```python 
    # Take only the top 5000 rows
    df = df.head(5000)
    ```
-  **Drop `image_url` Column**
    Since we are focusing on textual content, remove the image_url column.
    ```python
    # Drop the image_url column
    df = df.drop(columns=['image_url'])
    ```
### 3. Preprocessing Pipeline for Images

- **Convert Images to RGB**
    Ensure all images are in RGB format before embedding them.
    ```python
    from PIL import Image
    
    def convert_to_rgb(image_path):
        image = Image.open(image_path)
    return image.convert('RGB')

    # Convert images to RGB
    df['rgb_images'] = df['image_path'].apply(convert_to_rgb)
    ```
### 4.  Preprocessing Pipeline for Youtube
- **Filter Data**
    Filter the relevant fields: video id, publishedAt, title, and description.
    ```python
    video_data = [
        {
            'video_id': video['id']['videoId'],
            'publishedAt': video['snippet']['publishedAt'],
            'title': video['snippet']['title'],
            'description': video['snippet']['description']
        }
        for video in videos
    ]
    ```
### 5. Embedding Generation
-   **Text Embedding for Articles and Youtube**
    Model: We use `SentenceTransformer ('all-MiniLM-L6-v2')` for text embedding.
    ```python
    from sentence_transformers import SentenceTransformer

    # Load model for text embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for articles
    df['article_embedding'] = df['content'].apply(lambda x: model.encode(x))
    ```
- **Image Embedding for Images**
    Model: We use the `CLIP model (ViT-B/32)` to generate image embeddings. The device is set to CUDA if available, or CPU otherwise.
    ```python 
    import torch
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    def get_image_embedding(image):
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_tensor)
        return embedding.cpu().numpy()

    # Generate embeddings for images
    df['image_embedding'] = df['rgb_images'].apply(get_image_embedding)
    ```

### 6. Save the Embedding Vector
- After generating embeddings for articles, images and youtube, save the embedding vector.

    ```python
    import chromadb

    # Initialize the ChromaDB client
    client = chromadb.Client()

    # Create collections for articles, images, and YouTube videos
    article_collection = client.create_collection("article_collection")
    image_collection = client.create_collection("image_collection")
    youtube_collection = client.create_collection("youtube_collection")

    # Insert embeddings and data into the collections
    # Example for articles
    article_collection.add(
        ids=[f'article_{i}' for i in range(len(df_articles))],
        embeddings=df_articles['article_embedding'].tolist(),
        metadatas=df_articles[['title', 'description']].to_dict('records')
    )
    # Repeat for images and YouTube videos

    ```

## ⚙️ Setup and Running the Application
1. **Clone the Repository**
    ```bash
    git clone https://github.com/retno2777/BountyChromaDB.git
    cd BountyChromaDB
    ```
2. **Create Virtual Environment**
    ```bash
    python3 -m venv .venv
    .venv\Scripts\activate
    ```    
3. **Install Dependencies**<br />
Make sure to install the required dependencies before running the app.
    ```bash
    pip install -r requirements.txt
    ```
4. **Run app** <br />
Make sure run the command in `BountyChromaDB` directory and `venv` is active.
    ```bash
    python app.py
    ```
5. **Open The HTML Page** <br />
You can use extension **live server(optional)** or open the html in your browser.

6. **Enter Query**<br />
    Enter Your Query in form input then click submit.

### Image Overview 
1. **Text Query** 
![Text Query](https://github.com/retno2777/BountyChromaDB/blob/main/assetReadme/Text_search.png)

2. **Image Query**
![Image Query](https://github.com/retno2777/BountyChromaDB/blob/main/assetReadme/Image_search.png)

3. **Text Query Result**
![Text Query Result](https://github.com/retno2777/BountyChromaDB/blob/main/assetReadme/Text_result.png)

3. **Image Query Result**
![Image Query Result](https://github.com/retno2777/BountyChromaDB/blob/main/assetReadme/Image_result.png)

## Demo Video

![Demo Video](https://drive.google.com/file/d/1z_s3YmoIcOkAjzgW_3YHosTaq45JaET9/view?usp=sharing)


## 🚀 Enhancements

This section highlights key improvements that enhance the performance, accuracy, and overall optimization of the ChromaDB App.

### 1. Performance Metrics (🚀)

<ul>
  <li>
    <strong>Batch Processing:</strong>
    <p>Images are processed in batches to optimize memory usage and reduce latency. This allows efficient handling of large datasets by splitting the workload into smaller, manageable chunks.</p>
    <strong>Benefit:</strong> Reduces memory overhead and improves processing time when handling large amounts of data.
    <br/>
    <strong>Code Example:</strong>
    <pre><code>for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
    batch_images = image_files[i:i + batch_size]
    # Process batch...
    </code></pre>
  </li>
  
  <li>
    <strong>GPU Acceleration:</strong>
    <p>The app automatically detects and utilizes GPU acceleration (if available) to speed up the embedding generation process. This significantly reduces the time required to process images and text.</p>
    <strong>Benefit:</strong> Faster embedding generation by utilizing available GPU resources.
    <br/>
    <strong>Code Example:</strong>
    <pre><code>
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    </code></pre>
  </li>
</ul>

### 2. Accuracy Metrics (📊)

<ul>
  <li>
    <strong>Embedding Normalization:</strong>
    <p>After generating embeddings, vectors are normalized using <strong>L2 normalization</strong>. This step ensures that similarity searches using cosine distance are more accurate by ensuring the vectors have a consistent scale.</p>
    <strong>Benefit:</strong> Increases the accuracy of similarity searches.
    <br/>
    <strong>Code Example:</strong>
    <pre><code>image_embeds /= image_embeds.norm(dim=-1, keepdim=True)</code></pre>
  </li>
  
  <li>
    <strong>Flexible Distance Functions:</strong>
    <p>You can specify different distance functions (cosine, L2, or inner product) depending on the requirements of the similarity search. This flexibility provides better control over similarity calculations.</p>
    <strong>Benefit:</strong> Allows the user to adjust the distance metric to fit the embedding characteristics and improve search precision.
    <br/>
    <strong>Code Example:</strong>
    <pre><code>distance_function = "cosine"  # You can also use "l2" or "ip" search_results = article_collection.query(query_embeddings=[embedding], distance_metric=distance_function)
    </code></pre>
  </li>
</ul>

### 3. Optimizations (🛠)

<ul>
  <li>
    <strong>Memory Optimization:</strong>
    <p>Batch processing ensures only a subset of the data is loaded into memory at any given time. This is crucial when working with large datasets, as it prevents memory overload and optimizes resource usage.</p>
    <strong>Benefit:</strong> More efficient memory management, especially useful for large datasets.
    <br/>
    <strong>Code Example:</strong>
    <pre><code>batch_tensor = torch.stack(batch_images).to(device)
# Process the batch...
    </code></pre>
  </li>
  
  <li>
    <strong>Persistent Storage for Embeddings:</strong>
    <p>Embeddings are stored in ChromaDB with persistent storage, which allows the system to store embeddings and avoid recomputing them for future queries. This reduces the computational overhead for repeated or similar searches.</p>
    <strong>Benefit:</strong> Faster search performance due to persistent embeddings and reduced recomputation.
    <br/>
    <strong>Code Example:</strong>
    <pre><code>chroma_client = chromadb.PersistentClient(path=persist_directory)
    </code></pre>
  </li>
</ul>


### 🔑 Key Features of the Enhancements:
- **Batch processing** improves memory efficiency and allows scalable data processing.
- **GPU acceleration** significantly speeds up the embedding process when a GPU is available.
- **Normalization and flexible distance functions** increase the precision of similarity searches.
- **Persistent storage** ensures embeddings are stored for future use, reducing computational effort.

### ✨ Conclusion

The **ChromaDB App** exemplifies the power and flexibility of vector-based search technology, delivering efficient and accurate results across multiple content types. By utilizing advanced algorithms such as **Cosine Similarity** and **L2 Distance**, along with GPU acceleration and memory optimization techniques, this application offers an optimized and seamless experience for both **text** and **image searches**.

Participating in the **ChromaDB Bounty Quest** provided a valuable opportunity to push the boundaries of **ChromaDB** and vector search technology. This app demonstrates the capabilities of **ChromaDB** to handle complex, large-scale search tasks while maintaining performance and accuracy. With its intuitive interface and powerful backend, the **ChromaDB App** is well-suited for modern applications that demand speed and precision.

We are confident that this project lays a solid foundation for further innovations in the realm of vector search, and we look forward to exploring even more advanced use cases in the future.



