from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

# Load the trained model
model = torch.load("deepfake_cnn.pth", map_location=torch.device("cpu"))
model.eval()  # Set to evaluation mode

# Define preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match model input size
    transforms.ToTensor(),
])

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))  # Open image
    image = transform(image).unsqueeze(0)  # Apply transformations
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()  # Convert to probability

    result = "Deepfake" if prediction > 0.5 else "Real"
    return jsonify({"prediction": result, "confidence": prediction})

if __name__ == "__main__":
    app.run(debug=True)
