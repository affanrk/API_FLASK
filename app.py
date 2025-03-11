import torch
import numpy as np
import tempfile

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

@app.route('/', methods=['GET'])
@cross_origin()
def test():
    return 'FLASK RUNNING'

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # 1. Pastikan ada file dalam request
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada gambar dalam permintaan'}), 400

        image_file = request.files['image']
        
        # 2. Pastikan format gambar adalah JPG/JPEG
        if not image_file.filename.lower().endswith(('.jpg', '.jpeg')):
            return jsonify({'error': 'Format gambar harus JPG'}), 400

        # 3. Simpan gambar sementara
        temp_image = tempfile.NamedTemporaryFile(delete=False)
        image_file.save(temp_image.name)

        # # 4. Buka gambar dan cek ukuran
        img = Image.open(temp_image.name)

        # 5. Lakukan preprocessing
        img = transform(img).unsqueeze(0)

        # 6. Klasifikasi gambar
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.softmax(outputs, dim=1).numpy()

        # 7. Ambil hasil prediksi
        class_labels = {0: "Benchpress", 1: "Deadlift", 2: "Squat"}
        predicted_class_index = np.argmax(probabilities)
        predicted_class_label = class_labels[predicted_class_index]
        predicted_class_probability_percentage = "{:.2f}%".format(float(probabilities[0, predicted_class_index]) * 100)

        response = {
            'Class': predicted_class_label.capitalize(),
            'Prediction': predicted_class_label.capitalize(),
            'Probability': predicted_class_probability_percentage
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
