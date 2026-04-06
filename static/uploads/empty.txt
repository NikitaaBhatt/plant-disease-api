import os
from flask import Flask, redirect, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import gdown


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

MODEL_PATH = "plant_disease_model_1_latest.pt"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=10O1jKijNx09y4YdZ5YAw9AnzPDKzdZG-"
    print("Downloading model...")
    gdown.download(url, MODEL_PATH, quiet=False)

model = CNN.CNN(39)    
model.load_state_dict(
    torch.load("plant_disease_model_1_latest.pt", map_location=torch.device('cpu'))
)
model.eval()

def prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    input_data = TF.to_tensor(image).unsqueeze(0)

    input_data = input_data.float()   # 🔥 ensure correct dtype

    with torch.no_grad():
        output = model(input_data)

    output = output.cpu().numpy()
    index = np.argmax(output)

    return index

app = Flask(__name__)
CORS(app)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    image = request.files.get('image') or request.files.get('file')

    if image is None:
        return jsonify({'error': 'No image provided'}), 400

    if image.filename == '':
        return jsonify({'error': 'No selected image'}), 400
        
    filename = image.filename
    file_path = os.path.join('static/uploads', filename)
    os.makedirs('static/uploads', exist_ok=True)
    image.save(file_path)
    print("FILES RECEIVED:", request.files)
    print("IMAGE NAME:", image.filename)
    
    try:
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        
        return jsonify({
            'prediction': int(pred),
            'title': str(title),
            'description': str(description),
            'prevent': str(prevent),
            'image_url': str(image_url),
            'supplement_name': str(supplement_name),
            'supplement_image_url': str(supplement_image_url),
            'supplement_buy_link': str(supplement_buy_link)
        })
    except Exception as e:
        print("ERROR:", str(e))   # 🔥 IMPORTANT
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    os.makedirs('static/uploads', exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)