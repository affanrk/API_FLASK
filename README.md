# Flask API for Gym Exercise Classification

## Project Description
This project provides an API for classifying gym exercise images, specifically **Benchpress, Squat, and Deadlift**. The model used is **MobileViG**, a Graph Convolutional Network (GCN)-based architecture designed for image classification using **PyTorch**. 

For implementation, this API is built using **Flask** and serves as a bridge between the trained model and a website developed using the **Laravel** framework. The API enables real-time predictions based on user-uploaded images.

## Installation and Setup
### Prerequisites
Make sure you have **Python 3.10** installed.

### Install Dependencies
Run the following command to install all required dependencies:
```bash
pip install -r requirements.txt
```

### Running the API
To start the Flask server, execute:
```bash
python app.py
```
The API will be accessible at `http://0.0.0.0:5000`.

## API Endpoints
### 1. Test Endpoint
- **URL:** `/`
- **Method:** `GET`
- **Description:** Returns a confirmation message to indicate that the Flask server is running.
- **Response:**
```json
"FLASK RUNNING"
```

### 2. Prediction Endpoint
- **URL:** `/predict`
- **Method:** `POST`
- **Description:** Accepts an image file and returns the predicted exercise class along with the confidence score.
- **Request Format:**
  - `image`: A gym exercise image (Benchpress, Squat, or Deadlift)
  - `Content-Type: multipart/form-data`
- **Response Format:**
```json
{
  "Class": "Squat",
  "Prediction": "Squat",
  "Probability": "95.67%"
}
```

## Model Information
- **Model Used:** `MobileViG-Ti`
- **Trained Model File:** `m_MobileViG_ti_50_epoch_0.5_d.pth`
- **Input Image Size:** `224x224`
- **Transforms Applied:**
  - Resize to `(224, 224)`
  - Convert to tensor and normalize using ImageNet mean and std

## Folder Structure
```
├── model
│   ├── mobilevig.py  # Model architecture
│
├── app.py  # Flask API implementation
├── requirements.txt  # Dependencies list
├── README.md  # Documentation
```

Thank you for using this API! 🚀