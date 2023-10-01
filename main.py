# IMPORTS
from flask import Flask, render_template, request, url_for, session
import function_olivier as fo
import AI_module as ai
import pandas as pd
import openai
import os
from pathlib import Path
from PIL import Image
import time as time

openai.api_key = "API_KEY_HERE"

df = pd.read_csv('cleaned_data1.csv')
app=Flask(__name__)

UPLOAD_FOLDER = Path('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

info=[[0,"Pancakes", "Saturday Sep 30, 2023", "/static/0.jpg",250,8,8,37,7], [1,"Banana", "Friday Sep 29, 2023", "/static/1.jpg",110,0,1,28,15], [2,"Pizza", "Thursday Sep 28, 2023", "/static/2.jpg",285,10.4,12.2,35.7,3.8]]
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
        # print(image)

        # Do something with the uploaded image, e.g., save it or process it
        # return "uploaded"

        if image:
            # Ensure the 'static' folder exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            # Save the uploaded image to the 'static' folder
            image.filename= str(len(info)) + ".png"

            # image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
            

            filename = app.config['UPLOAD_FOLDER'] / image.filename
            image.save(filename)

            img = Image.open(filename)
            img = ai.prep_img(img)

            model = ai.load_resnet("train_AI/models_weights/test_resnet_18_256.pth")

            pred = ai.prediction(model, img)

            word = fo.getindexofmaxofnparray(pred)
            data = fo.search_reviews(df, word, n=1, pprint=True)

            #img_url=url_for('static', filename=f'uploads/{image.filename}')
            #turn pd dataframe row into list and add to new list
            new_list = data.values.tolist()
            new_list = new_list[0]
            new_list.pop(len(new_list) - 1)
            new_list.insert(0, len(info))
            #append date to list in position 3 (index 2)
            new_list.insert(2, time.strftime("%A %b %d, %Y"))
            #append filename to list in position 4 (index 3)
            new_list.insert(3, filename)
            # info.insert(0,[len(info),"grapes","saturday 12 2023",filename,1,2,3,4,5])
            info.reverse()
            info.append(new_list)
        return render_template('index.html', info=info, selected_item=info[button_clicked])
    else:
        return 'No image file uploaded'
    
if __name__ == '__main__':
    app.run(debug=True)