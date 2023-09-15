from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import base64
import io
from category_list import category_names

# Initialize the FastAPI app
app = FastAPI()

# Mount the "static" directory for serving static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2Templates for rendering HTML templates
templates = Jinja2Templates(directory="app/templates")

# Define the URL for the MobileNetV2 model from TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

# Load the MobileNetV2 model using TensorFlow Hub
model = tf.keras.Sequential([hub.KerasLayer(model_url)])

# Function to preprocess an uploaded image
def preprocess_image(file, target_size=(224, 224)):
    img = Image.open(file.file)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    if img_array.shape[-1] != 3:
        img_array = np.stack((img_array,) * 3, axis=-1)
    return img_array

# Create a dictionary to map class indices to category names
class_index_to_category_name = {i: category_names[i] for i in range(len(category_names))}

# Function to predict the category of an image
def predict_category(image):
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_category = np.argmax(predictions, axis=1)
    confidence_score = predictions[0][predicted_category]
    return predicted_category, confidence_score

# Function to convert an image to base64 format
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Route for predicting the category of an uploaded image
@app.post("/predict/")
async def predict_category_route(request: Request, file: UploadFile = File(...)):
    # Convert the image to bytes
    image = Image.open(io.BytesIO(await file.read()))
    image_data_uri = f"data:image/png;base64,{image_to_base64(image)}"

    # Preprocess the image and make a prediction
    preprocessed_image = preprocess_image(file)
    predicted_category, confidence_score = predict_category(preprocessed_image)

    # Render the HTML template with the image data URI and prediction data
    return templates.TemplateResponse("index.html", {
        "request": request,
        "image_data_uri": image_data_uri,
        "category": predicted_category[0],
        "category_name": class_index_to_category_name[predicted_category[0]],
        "confidence": confidence_score[0]
    })

# Default route
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "category": None, "confidence": None})

# Entry point for running the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
