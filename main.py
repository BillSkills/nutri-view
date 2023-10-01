from flask import Flask, render_template, request, url_for
import function_olivier as fo
import pandas as pd
import openai
import os
from pathlib import Path


openai.api_key = "INSERT HERE"

df = pd.read_csv('cleaned_data1.csv')
app=Flask(__name__)

UPLOAD_FOLDER = Path('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=["GET","POST"])
def index():
    #info=fo.search_reviews(df, "porridge", n=1, pprint=True)
    info=[[0,"banana", "wednesday 12 2023", 1,2,3,4,5],[1,"apple", "tuesday 13 2023", 1,2,3,4,5],[2,"pear", "thursday 12 2023", 1,2,3,4,5],[3,"strawberry", "friday 12 2023", 1,2,3,4,5]]
    selected_item=None

    if request.method=="POST":
        button_clicked=request.form['button_clicked']
        selected_item = info[int(button_clicked)]
        print(button_clicked)
        print(selected_item)

    if 'image' not in request.files:
        print("I hate you")
    else:
        image = request.files['image']
        print(image)
        if image:
            filename = app.config['UPLOAD_FOLDER'] / image.filename
            image.save(filename)
            img_url=url_for('static', filename=f'uploads/{image.filename}')

    
    return render_template('index.html', info=info, selected_item=selected_item)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        # Do something with the uploaded image, e.g., save it or process it
        return 'Image uploaded successfully'
    else:
        return 'No image file uploaded'
    
if __name__ == '__main__':
    app.run(debug=True)