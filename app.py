import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image

app = FastAPI(title="Edge-AI Crop Disease Detector")

MODEL_PATH = "models/crop_model.tflite"
CLASSES = ["Early_Blight", "Healthy", "Late_Blight"]

# Action Plans & Cures DB
TREATMENT_ADVICE = {
    "Early_Blight": {
        "symptoms": "Dark brown spots with concentric rings on leaves.",
        "treatment": "Apply copper-based fungicides immediately. Prune infected lower leaves to prevent spore spread.",
        "prevention": "Ensure proper spacing for airflow and avoid overhead watering."
    },
    "Late_Blight": {
        "symptoms": "Water-soaked dark lesions on leaves and stems with white mold growth underneath.",
        "treatment": "Spray systemic fungicides like Mancozeb or Chlorothalonil. Remove severely infected crops.",
        "prevention": "Use disease-resistant seed varieties and avoid high humidity pooling."
    },
    "Healthy": {
        "symptoms": "No visible leaf spots, discoloration, or fungal lesions.",
        "treatment": "No chemical treatment required.",
        "prevention": "Maintain regular drip irrigation, balanced NPK fertilizing, and soil testing."
    }
}

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Edge-AI Crop Health Assistant</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            body { background: #f0f4f1; font-family: 'Inter', system-ui, sans-serif; min-height: 100vh; }
            .hero-card { background: #ffffff; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); border: none; }
            .drop-zone { border: 2px dashed #2e7d32; border-radius: 15px; background: #f8fcf9; transition: all 0.3s ease; cursor: pointer; }
            .drop-zone:hover { background: #e8f5e9; border-color: #1b5e20; }
            .btn-custom { background-color: #2e7d32; color: white; border-radius: 10px; font-weight: 600; padding: 12px 30px; transition: 0.3s; }
            .btn-custom:hover { background-color: #1b5e20; color: white; transform: translateY(-2px); }
            .result-card { border-radius: 15px; border-left: 6px solid #2e7d32; display: none; }
            #preview { max-height: 220px; border-radius: 12px; display: none; margin: 0 auto; object-fit: cover; }
            .badge-custom { font-size: 0.9rem; padding: 8px 16px; border-radius: 20px; }
        </style>
    </head>
    <body>
        <div class="container py-5">
            <div class="row justify-content-center">
                <div class="col-lg-8">
                    <div class="hero-card p-4 p-md-5">
                        <div class="text-center mb-4">
                            <span class="badge bg-success-subtle text-success badge-custom mb-2"><i class="fa-solid fa-microchip me-1"></i> Edge-AI Powered</span>
                            <h2 class="fw-bold text-dark">Crop Disease Diagnostic Portal</h2>
                            <p class="text-muted">Upload leaf image for instant AI diagnosis and agricultural treatment guidelines.</p>
                        </div>

                        <!-- Upload Area -->
                        <div class="drop-zone p-4 text-center mb-4" onclick="document.getElementById('imageInput').click()">
                            <i class="fa-solid fa-cloud-arrow-up fa-3x text-success mb-3"></i>
                            <h6 class="fw-bold text-dark">Click to upload or drag & drop leaf scan</h6>
                            <p class="text-muted small mb-0">Supports JPG, PNG formats</p>
                            <input type="file" id="imageInput" class="d-none" accept="image/*" onchange="previewImage(event)">
                        </div>

                        <!-- Image Preview -->
                        <div class="text-center mb-4">
                            <img id="preview" alt="Leaf Scan" class="img-fluid shadow-sm">
                        </div>

                        <div class="text-center">
                            <button class="btn btn-custom w-100" onclick="uploadImage()">
                                <i class="fa-solid fa-stethoscope me-2"></i>Analyze Crop Health
                            </button>
                        </div>

                        <!-- Results Dashboard -->
                        <div id="resultCard" class="card result-card mt-5 p-4 bg-light shadow-sm">
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <h4 class="fw-bold mb-0 text-dark" id="resTitle">Diagnosis Result</h4>
                                <span class="badge bg-success text-white" id="resScore">98% Accuracy</span>
                            </div>
                            
                            <hr>

                            <div class="row g-3">
                                <div class="col-12">
                                    <h6 class="fw-bold text-success"><i class="fa-solid fa-triangle-exclamation me-2"></i>Symptoms</h6>
                                    <p id="resSymptoms" class="text-muted small mb-2"></p>
                                </div>
                                <div class="col-12">
                                    <h6 class="fw-bold text-danger"><i class="fa-solid fa-kit-medical me-2"></i>Recommended Treatment</h6>
                                    <p id="resTreatment" class="text-muted small mb-2"></p>
                                </div>
                                <div class="col-12">
                                    <h6 class="fw-bold text-primary"><i class="fa-solid fa-shield-halved me-2"></i>Prevention Advice</h6>
                                    <p id="resPrevention" class="text-muted small mb-0"></p>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </div>

        <script>
            function previewImage(event) {
                const img = document.getElementById('preview');
                if (event.target.files[0]) {
                    img.src = URL.createObjectURL(event.target.files[0]);
                    img.style.display = 'block';
                }
            }

            async function uploadImage() {
                const input = document.getElementById('imageInput');
                if (!input.files[0]) {
                    alert('Please select an image file first.');
                    return;
                }

                const formData = new FormData();
                formData.append('file', input.files[0]);

                try {
                    const response = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await response.json();

                    document.getElementById('resTitle').innerText = data.predicted_disease.replace('_', ' ');
                    document.getElementById('resScore').innerText = (data.confidence_score * 100).toFixed(1) + '% Confidence';
                    
                    document.getElementById('resSymptoms').innerText = data.details.symptoms;
                    document.getElementById('resTreatment').innerText = data.details.treatment;
                    document.getElementById('resPrevention').innerText = data.details.prevention;

                    document.getElementById('resultCard').style.display = 'block';
                } catch (err) {
                    alert('Analysis failed. Make sure server is running.');
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    contents = await file.read()
    input_data = preprocess_image(contents)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_idx = int(np.argmax(output_data[0]))
    confidence = float(np.max(output_data[0]))
    
    predicted_disease = CLASSES[predicted_class_idx]
    details = TREATMENT_ADVICE.get(predicted_disease, TREATMENT_ADVICE["Healthy"])

    return {
        "status": "success",
        "predicted_disease": predicted_disease,
        "confidence_score": round(confidence, 4),
        "details": details
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)