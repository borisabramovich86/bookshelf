from flask import Flask, request, jsonify
from analyze_and_recommend import analyze_and_recommend
import logging
from PIL import Image
import io

logging.getLogger().setLevel(logging.INFO)

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend_books_from_image():
    ocr = request.args.get('ocr') 
    to_visualize = request.args.get('visualize')

    logging.info('Got recommendation request')

    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))

    processor_type = ocr or "ollama"
    recommender_type = "ollama"
    visualize = bool(to_visualize) or False

    results = analyze_and_recommend(image, processor_type, recommender_type, visualize=visualize, save_to_file=False)
    json_results = jsonify(results.model_dump()["books"])
    return json_results

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"ok": True})

if __name__ == '__main__':
    app.logger.setLevel(logging.INFO)
    app.run(port=5000)