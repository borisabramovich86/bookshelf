from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    # Save the image to a folder (e.g., "uploads")
    image.save(f"uploads/{image.filename}")

    return jsonify({'message': f"Image '{image.filename}' uploaded successfully!"})

if __name__ == '__main__':
    app.run(debug=True)