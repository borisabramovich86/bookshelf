from flask import Flask, request, jsonify
from analyze_and_recommend import analyze_and_recommend
import logging

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend_books_from_image():
    logging.info('Got recommendation request')
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    image = request.files['image']

    processor_type = "ollama"
    recommender_type = "ollama"

    results = analyze_and_recommend(image, processor_type, recommender_type, visualize=False, save_to_file=False)
    return results.json()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"ok": True})

if __name__ == '__main__':
    app.run(debug=True)