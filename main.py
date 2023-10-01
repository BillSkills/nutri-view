# IMPORTS
from flask import Flask, render_template, request, url_for, session
import function_olivier as fo
import pandas as pd
import openai
import os
from pathlib import Path
import shutil
from PIL import Image

openai.api_key = "INSERT HERE"

df = pd.read_csv('cleaned_data1.csv')
app=Flask(__name__)

UPLOAD_FOLDER = Path('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

info=[[0,"banana", "tuesday 12 2023", "static/0.png",1,2,3,4,5]]
info.reverse()
button_clicked = 0

@app.route('/', methods=["GET","POST"])
def index():
    #info=fo.search_reviews(df, "porridge", n=1, pprint=True)
    selected_item=None

    if request.method=="POST":
        button_clicked=int(request.form['button_clicked'])
        selected_item = info[button_clicked]
        print(button_clicked)
        print(selected_item)

    if 'image' not in request.files:
        print("Image not in request.files")

    
    return render_template('index.html', info=info, selected_item=selected_item)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        print(image)

        # Do something with the uploaded image, e.g., save it or process it
        # return "uploaded"

        if image:
            # Ensure the 'static' folder exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            # Save the uploaded image to the 'static' folder
            image.filename= str(button_clicked) + ".png"

            # image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
            

            filename = app.config['UPLOAD_FOLDER'] / image.filename
            image.save(filename)

            info.insert(0,[len(info),"grapes","saturday 12 2023",filename,1,2,3,4,5])
            #img_url=url_for('static', filename=f'uploads/{image.filename}')

        return render_template('index.html', info=info, selected_item=info[button_clicked])
    else:
        return 'No image file uploaded'
    
if __name__ == '__main__':
    app.run(debug=True)