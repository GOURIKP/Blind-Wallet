from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import joblib
import cv2
import os
import sys
import pickle
from skimage import feature
import io
import pyttsx3

app = Flask(__name__)

# Configure file paths and parameters
BOVW="bovw_codebook_600.pickle"
MODEL='rfclassifier_600.sav'
IMG_SIZE=320

# Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# Local Binary Patterns (LBP) Texture
def fd_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = feature.local_binary_pattern(gray, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Color Histogram
def fd_histogram(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bins=8
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# SIFT Bag of Visual Words
def feature_extract(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    feature = bowDiction.compute(gray, sift.detect(gray))
    return feature.squeeze()

# Load the BOVW codebook
pickle_in = open(BOVW,"rb")
dictionary = pickle.load(pickle_in)

# Initialize SIFT BOW image descriptor extractor
sift2 = cv2.xfeatures2d.SIFT_create()
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)

# Load the trained model
loaded_model = joblib.load(MODEL)

# Class-label dictionary
label= {0:"10", 1:"20", 2:"50", 3:"100", 4:"200", 5:"500" , 6:"2000"}

def preprocess_image(image):
    # Convert PIL Image to OpenCV format
    open_cv_image = np.array(image)
    image = open_cv_image[:, :, ::-1].copy() 
    
    # Resize the image
    (height, width, channel) = image.shape
    resize_ratio = 1.0 * (IMG_SIZE / max(width, height))
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    input_image = cv2.resize(image, target_size)
    
    # Preprocess the image using the functions from prediction script
    Hu=fd_hu_moments(input_image)
    LBP=fd_lbp(input_image)
    Hist=fd_histogram(input_image)
    Bovw=feature_extract(input_image)
    
    # Generate a feature vector by combining all features
    mfeature= np.hstack([Hu, LBP, Hist, Bovw])
    
    return mfeature

# Define an API endpoint to handle image classification
@app.route('/classify_currency', methods=['POST'])
def classify_currency():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    img = Image.open(io.BytesIO(image.read()))
    
    try:
        processed_image = preprocess_image(img)
        prediction = loaded_model.predict(processed_image.reshape(1, -1))
        class_label = prediction[0]  # Assuming the class label is directly returned by the model
        currency_labels = ['10 Rupees', '20 Rupees', '50 Rupees', '100 Rupees', '200 Rupees', '500 Rupees', '2000 Rupees']
        result = {'currency': currency_labels[class_label]}

        # Speak the result
        engine = pyttsx3.init()
        engine.say("The detected currency is " + currency_labels[class_label])
        engine.runAndWait()
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
