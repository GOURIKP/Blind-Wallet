import numpy as np
import cv2
import os
import sys
import pickle
import joblib
from skimage import feature


# Configure file paths and parameters
IMAGE=sys.argv[1]
BOVW="model/bovw_codebook_600.pickle"
MODEL='model/rfclassifier_600.sav'
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

# Load the trained model and input image
loaded_model = joblib.load(MODEL)
image=cv2.imread(IMAGE)

# Resize the image
(height, width, channel) = image.shape
resize_ratio = 1.0 * (IMG_SIZE / max(width, height))
target_size = (int(resize_ratio * width), int(resize_ratio * height))
input_image = cv2.resize(image, target_size)

cv2.imwrite("res_img.png",input_image)

# Class-label dictionary
label= {0:"10", 1:"20", 2:"50", 3:"100", 4:"200", 5:"500" , 6:"2000"}

# Load the BOVW codebook
pickle_in = open(BOVW,"rb")
dictionary = pickle.load(pickle_in)

# Initialize SIFT BOW image descriptor extractor
sift2 = cv2.xfeatures2d.SIFT_create()
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)

# Extract the features
Hu=fd_hu_moments(input_image)
LBP=fd_lbp(input_image)
Hist=fd_histogram(input_image)
Bovw=feature_extract(input_image)

# Generate a feature vector by combining all features
mfeature= np.hstack([Hu, LBP, Hist, Bovw])

# Predict the output using trained model
output = loaded_model.predict(mfeature.reshape((1,-1)))
print("\nPredicted class: "+label[output[0]])

'''
Sample run: python predict.py test/2000.jpg
'''
