import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score 
import joblib
import h5py

# Set default model paths
DEFAULT_MODEL_SAV = 'model/rfclassifier_600.sav'
DEFAULT_MODEL_H5 = 'model/rfclassifier_600.h5'

# Get model paths from command-line arguments or use defaults
if len(sys.argv) > 2:
    MODEL_SAV = sys.argv[1]
    MODEL_H5 = sys.argv[2]
else:
    MODEL_SAV = DEFAULT_MODEL_SAV
    MODEL_H5 = DEFAULT_MODEL_H5

# Configure file paths
DATA='model/data_600.npy'
LABEL='model/label_600.npy'

# Load data and labels
data=np.load(DATA)
label=np.load(LABEL)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=42)

# Initialize a random forest classifier
print("\n<====Model====>\n")
clf = RandomForestClassifier(n_estimators=200,n_jobs=-1,verbose=1)
print(clf)

print("\n<====Training====>\n")

# Train the model
clf.fit(x_train, y_train)

# Check accuracy on test data
y_pred = clf.predict(x_test)
y_true = y_test
print("\nAccuracy:", accuracy_score(y_true,y_pred))

# Compute the confusion matrix
print("\n<====Confusion Matrix====>\n")
cf=confusion_matrix(y_true, y_pred)
print(cf)

print("\n<====Cross validation====>\n")

# Evaluate the model using five fold cross validation
scores = cross_val_score(clf, x_train, y_train, cv=5)  
print("\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Save the trained model in .sav format
joblib.dump(clf, MODEL_SAV)

# Save the trained model in .h5 format
with h5py.File(MODEL_H5, 'w') as hf:
    hf.create_dataset('model', data=clf)

'''
Sample run: python train.py model/rfclassifier_600.sav model/rfclassifier_600.h5
'''
