#Flask and Json
from flask import Flask, jsonify, render_template, url_for, request
import json
import requests
import os
import cv2
from imageai.Prediction.Custom import CustomImagePrediction
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

#Flask Config ve Context Procssing(Cachleme işlemi)
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # json ascıı
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True # json veri döndürmek için.
port = int(os.environ.get("PORT", 5000)) # herokuapp port ayarları.



@app.route("/")
def test():

    return "test api"



@app.route("/api/<path:url>", methods=['GET'])
def ss(url):
    
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    img.save("test.jpg", "jpeg")
    img = cv2.imread("test.jpg")
    os.remove("test.jpg")

    '''
    #percent by which the image is resized
    scale_percent = 50

    #calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)
    # resize image
    img = cv2.resize(img, dsize)
    '''


    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath("./model/model_ex-056_acc-1.000000.h5")
    prediction.setJsonPath("./model/model_class.json")
    prediction.loadModel(num_objects=2)


    predictions, probabilities = prediction.predictImage(input_type="array", image_input=img, result_count=2)
    '''
    predictions = ['duvar', 'kahve']
    probabilities = [99.000223, 0.79023123]

    [('duvar', 99.000223), ('kahve', 0.79023123)]
    '''
    
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        if eachPrediction == "kahve":
            if eachProbability < 0.30:
                json_buffer = {"result": False, "percent": str(eachProbability)}
            
            else:
                json_buffer = {"result": True, "percent": str(eachProbability)}
        else:
            json_buffer = {"result": False, "percent": str(eachProbability)}           
    
    return jsonify(json_buffer)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)



