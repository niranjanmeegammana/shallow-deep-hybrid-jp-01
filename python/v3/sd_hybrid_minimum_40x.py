import psutil
import time
from keras.models import load_model
import os
import numpy as np

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

current_folder = "d:\\miniconda\\UNSW-NB15\\testing"
slash="\\"
split="_90_5_5b_"
datapath=current_folder +slash + "data" + slash
modelpath=current_folder + slash + "hy_models" + slash

#f_Xtest20=datapath + 'unsw-nb15_Xtest20' + split+ '.csv'
f_Xtest40=datapath + 'unsw-nb15_Xtest40' + split+ '.csv'
f_ytest=datapath + 'unsw-nb15_ytest' + split+ '.csv'

#-------------------
xmodel="shadeep_hyb_minimum_40"
#X_test = np.loadtxt(f_Xtest20, delimiter=',')
X_test = np.loadtxt(f_Xtest40, delimiter=',')
y_test = np.loadtxt(f_ytest, delimiter=',')
#-------------------
f_performance_file=datapath + xmodel+ '1x.txt'
model_path=modelpath+ xmodel+ '_model.keras'

#print(f_Xtest20)
print(f_Xtest40)
print(f_ytest)
print(model_path)
print("Running Testing with full data set")

# Measure CPU and memory usage before prediction
cpu_usage_before = psutil.cpu_percent()
memory_usage_before = psutil.virtual_memory().used

# Load the saved model
saved_model = load_model(model_path)

# Record start time for prediction
start_time = time.time()

# Predict on the test data
y_pred = saved_model.predict(X_test)

# Convert predicted probabilities to binary preds(0 or 1)
y_pred_classes = (y_pred > 0.5).astype(int)

# Record end time for prediction
end_time = time.time()

# Measure CPU and memory usage after prediction
cpu_usage_after = psutil.cpu_percent()
memory_usage_after = psutil.virtual_memory().used

# Calculate prediction time
prediction_time = end_time - start_time
# Print CPU and memory usage, and prediction time
print("CPU usage before prediction:", cpu_usage_before)
print("Memory usage before prediction:", memory_usage_before)
print("CPU usage after prediction:", cpu_usage_after)
print("Memory usage after prediction:", memory_usage_after)
print("Prediction time:", prediction_time)

# Define the file path where you want to save the output
output_file_path = f_performance_file

# Open the file in write mode
with open(output_file_path, 'w') as file:
    # Write performance metrics to the file
    file.write(f"{xmodel} Resource Use :Full Dataset: \n")
    file.write(f"CPU Usage Before Prediction: {cpu_usage_before}\n")
    file.write(f"Memory Usage Before Prediction: {memory_usage_before}\n")
    file.write(f"CPU Usage After Prediction: {cpu_usage_after}\n")
    file.write(f"Memory Usage After Prediction: {memory_usage_after}\n")
    file.write(f"Prediction Time: {prediction_time} seconds\n")

# Print a message indicating that the output was saved
print(f"Performance metrics and system information saved to: {output_file_path}")