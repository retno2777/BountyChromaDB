from flask import Flask, request, jsonify
from util_config.query_utils_string import query_all_collections
from util_config.query_utils_image import query_all_collections_with_image
import time
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Menangani preflight OPTIONS request
@app.before_request
def handle_options_request():
    if request.method == 'OPTIONS':
        # Mengembalikan response dengan header CORS yang tepat
        response = app.make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response

@app.route('/search', methods=['POST'])
def search():
    # Initialize search_results variable
    search_results = None

    # Start counting time before query
    start_time = time.time()

    # If there is an image file uploaded, perform search by image
    if 'image' in request.files:
        image_file = request.files['image']
        query_text = 'image query'
        distance_type = request.form.get('distance_type', 'cosine')

        # Make sure the file is an image
        if image_file and allowed_file(image_file.filename):
            # Save the image temporarily in the uploads folder
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Call query_all_collections_with_image function with the image
            search_results = query_all_collections_with_image(image_path, top_k=1, distance_type=distance_type)

            # Delete the image after it's used to save space
            os.remove(image_path)
        else:
            return jsonify({'error': 'Invalid file type. Only images are allowed.'}), 400

    # If there is no image, check if there is a query text
    else:
        query_text = request.form.get('query')
        distance_type = request.form.get('distance_type', 'cosine')  # Default to 'cosine' if not specified

        # Make sure query text is available
        if not query_text:
            return jsonify({'error': 'Query text or image is required'}), 400

        # Call query_all_collections function with the query text
        search_results = query_all_collections(query_text, top_k=1, distance_type=distance_type)

        # If no results are found, return an appropriate response
        if not search_results:
            return jsonify({'error': f"No results found for query: '{query_text}'"}), 404

    # Calculate the time spent on the query
    query_time = time.time() - start_time

    # Create a JSON response with the results and query time
    response_data = {
        'query': query_text,
        'distance_type': distance_type,
        'query_time': round(query_time, 4),
        'results': search_results
    }

    # Return the query results in JSON format, including the query time
    response = jsonify(response_data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response, 200

def allowed_file(filename):
    """
    Memeriksa apakah file yang diunggah adalah gambar berdasarkan ekstensi file.
    """
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    app.run(debug=True)
