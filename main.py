# from flask import Flask, render_template, request
from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from pathlib import Path


app = FastAPI()


BASE_DIR = Path(__file__).resolve(strict=True).parent
model = torch.load(f"{BASE_DIR}/Entire_resnet50_model.pth",
                   map_location=torch.device('cpu'))
model.eval()


@app.route('/predict', methods=['POST'])
def predict(path):
    data_transforms = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #file = request.files['file']
    # Preprocess the image using the data_transforms pipeline
    img = Image.open(path)
    img = data_transforms(img)
    img = img.unsqueeze(0)  # Add batch dimension
    # Make a prediction using the pre-trained model
    with torch.no_grad():
        output = model(img)
        # Convert the output to a numpy array
        output_np = output.numpy()
        # Convert the output to a list of class probabilities
        probs = np.exp(output_np) / np.sum(np.exp(output_np), axis=1)
        # ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']
        classes = ['Branchiomycosis', 'argulus', 'myxobolosis']
        class_probs = {class_name: prob.item()
                       for class_name, prob in zip(classes, probs[0])}
        print(class_probs)
    # img.close() # Close the file
    # Return the predicted class as a JSON response
    return class_probs


@app.get("/")
def home():
    return {"healthCheck": "OK"}


@app.post("/submit")
async def get_output(file: UploadFile):
    # if request.method == 'POST':
    # img = file['my_image']

    # img = payload.text
    img_path = "static/" + file.filename
    contents = await file.read()
    with open(f"{BASE_DIR}/img_path", "wb") as f:
        f.write(contents)
    # file.save(img_path)

    p = predict(f"{BASE_DIR}/img_path")
    predicted_class = max(p, key=p.get)
    print(predicted_class)

    return {'response': predicted_class}
    # return render_template("index.html", prediction=predicted_class, img_path=img_path)


# if __name__ == '__main__':
#     app.run(debug=True)
