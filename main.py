import torch
import numpy as np
import tempfile
import timm

from flask import Flask, request, jsonify
from torchvision.transforms import v2
from flask_cors import CORS, cross_origin
from model import mobilevig
from PIL import Image

app = Flask(__name__)
cors = cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load the PyTorch model
model = mobilevig.mobilevig_ti(num_classes=3)
model.load_state_dict(torch.load('m_MobileViG_ti_50_epoch_0.5_d.pth', map_location=torch.device('cpu'), weights_only=False))
model.eval()  # Set the model to evaluation mode

# Define the image transformation pipeline (resize, normalize, etc.)
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/test', methods=['GET'])
@cross_origin()
def test():
    return jsonify({'FLASK RUNNING'})

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # Check if 'image' is in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image in the request'}), 400

        # Save the image temporarily
        image_file = request.files['image']
        temp_image = tempfile.NamedTemporaryFile(delete=False)
        image_file.save(temp_image.name)

        # Open the image and apply transformations
        img = Image.open(temp_image.name)
        img = transform(img).unsqueeze(0)  # Add batch dimension

        # Run the image through the model
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.softmax(outputs, dim=1).numpy()

        # Define class labels
        waste_labels = {0: "Benchpress", 1: "Deadlift", 2: "Squat"}

        # Get the predicted class
        predicted_class_index = np.argmax(probabilities)
        predicted_class_label = waste_labels[predicted_class_index]
        predicted_class_probability_percentage = "{:.2f}%".format(float(probabilities[0, predicted_class_index]) * 100)

        # Build the response
        response = {
            'Class': predicted_class_label.capitalize(),
            'Prediction': predicted_class_label.capitalize(),
            'Probability': predicted_class_probability_percentage
        }

        print(response)
        return jsonify(response)

    except Exception as e:
        print('An error occurred:', str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
