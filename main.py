from flask import Flask, render_template, request
import function_olivier as fo
import pandas as pd
import openai

openai.api_key = "INSERT HERE"

df = pd.read_csv('cleaned_data1.csv')
app=Flask(__name__)
@app.route('/')
def index():
    info=fo.search_reviews(df, "porridge", n=1, pprint=True)
    
    return render_template('testfile.html', info=info)

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