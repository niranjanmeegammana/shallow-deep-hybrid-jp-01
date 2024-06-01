import psutil
import time
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l1, l2
#from kerastuner import HyperParameters
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

xmodel="shallow_20"
# Define the features
features = ['ct_state_ttl', 'sload', 'rate', 'sttl', 'smean', 'dload', 'sbytes', 'ct_srv_dst', 'ct_dst_src_ltm', 'dbytes', 'ackdat', 'dttl', 'ct_dst_sport_ltm', 'dmean','ct_srv_src', 'dinpkt', 'tcprtt', 'dur', 'synack', 'sinpkt']

current_folder = "d:\\miniconda\\UNSW-NB15\\testing"
print(current_folder)
slash="\\"
split="_90_5_5b_"
datapath=current_folder +slash + "data" + slash
modelpath=current_folder + slash + "models" + slash
f_train=datapath+'unsw-nb15_training' + split +'.csv'
f_test=datapath + 'unsw-nb15_testing' + split+ '.csv'
f_val=datapath + 'unsw-nb15_validation' + split+ '.csv'

f_performance_file=datapath + xmodel+ '.txt'

model_path=modelpath+ xmodel+ '_model_ANN5_hp.keras'

print(f_test)
print(f_train)
print(f_val)
print(model_path)

df_train = pd.read_csv(f_train)
df_test = pd.read_csv(f_test)
df_val=pd.read_csv(f_val)
print(df_test.head(5))

# Separate features and labels for training set
X_train = df_train[features].values # assuming 'label' is the column containing labels
y_train = df_train['label'].values

# Separate features and labels for testing set
X_test = df_test[features].values  # assuming 'label' is the column containing labels
y_test = df_test['label'].values

# Separate features and labels for testing set
X_val = df_val[features].values  # assuming 'label' is the column containing labels
y_val = df_val['label'].values


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

#==================================

# Define a learning rate scheduler function
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def build_model(hp): # shallow
    model = Sequential()

    # Input layer
    model.add(Dense(units=512,
                    input_dim=len(features),
                    activation=hp['activation_layer1'],
                    kernel_initializer=hp['weight_initializer'],
                    kernel_regularizer=l1(hp['regularization_strength']) if hp['regularization_type'] == 'l1' else
                                     l2(hp['regularization_strength']) if hp['regularization_type'] == 'l2' else None))

    # Dropout layer (optional)
    if hp['use_dropout']:
        model.add(Dropout(rate=hp['dropout_rate']))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Define the learning rate scheduler
    lr_schedule = LearningRateScheduler(lr_scheduler)

    # Choose the optimizer based on the hyperparameter value
    optimizer = None
    if hp['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp['learning_rate'])
    elif hp['optimizer'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp['learning_rate'])
    elif hp['optimizer'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=hp['learning_rate'], momentum=0.9)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Example usage:
# Assuming 'best_hyperparameters' is a dictionary containing the best hyperparameters from random search

best_hyperparameters = {
    'activation_layer1': 'relu',
    'optimizer': 'rmsprop',
    'learning_rate': 0.001,
    'weight_initializer': 'he_normal',
    'batch_size': 256,
    'regularization_type': '',
    'regularization_strength': 0.01,
    'use_lr_scheduler': 0,
    'use_dropout': 0,
    'dropout_rate': 0.3
}
epochs_to_run=200

#============================
# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)


# Define the filepath to save the model
checkpoint_filepath = model_path

# Define a ModelCheckpoint callback to save the model with the highest validation accuracy
checkpoint = ModelCheckpoint(checkpoint_filepath,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)
                             
                             # Record start time for prediction
start_time = time.time()
# Build the shallow ANN model using the best hyperparameters
ann_model = build_model(best_hyperparameters)

# Train the model with early stopping and ModelCheckpoint callback
history = ann_model.fit(X_train, y_train,
                        epochs=epochs_to_run,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, checkpoint])

# Calculate training time
training_time = time.time() - start_time
                       
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']

# Access validation history
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

max_val_acc_epoch = np.argmax(history.history['val_accuracy'])
max_val_acc = history.history['val_accuracy'][max_val_acc_epoch]
min_val_loss_epoch = np.argmin(history.history['val_loss'])
min_val_loss = history.history['val_loss'][min_val_loss_epoch]

print("Epoch with maximum validation accuracy:", max_val_acc_epoch)
print("Maximum validation accuracy:", max_val_acc)
print("Epoch with minimum validation loss:", min_val_loss_epoch)
print("Minimum validation loss:", min_val_loss)

# Save the shallow ANN model to model_path
ann_model.save(model_path)

print(model_path, " saved")
# Print model summary
ann_model.summary()


# Load the saved model
saved_model = load_model(model_path)

# Measure CPU and memory usage before prediction
cpu_usage_before = psutil.cpu_percent()
memory_usage_before = psutil.virtual_memory().used

# Record start time for prediction
start_time = time.time()

# Assuming X_test and y_test are your test data
# Predict on the test data
y_pred = saved_model.predict(X_test)

# Convert predicted probabilities to binary predictions (0 or 1)
y_pred_classes = (y_pred > 0.5).astype(int)

# Record end time for prediction
end_time = time.time()

# Measure CPU and memory usage after prediction
cpu_usage_after = psutil.cpu_percent()
memory_usage_after = psutil.virtual_memory().used

# Calculate prediction time
prediction_time = end_time - start_time

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)
roc_auc = roc_auc_score(y_test, y_pred)
avg_precision = average_precision_score(y_test, y_pred)

# Compute confusion matrix
conf_matrix = tf.math.confusion_matrix(y_test, y_pred_classes).numpy()

# Print performance metrics
print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Average Precision:", avg_precision)
print("Confusion Matrix:\n", conf_matrix)

# Print CPU and memory usage, and prediction time
print("CPU usage before prediction:", cpu_usage_before)
print("Memory usage before prediction:", memory_usage_before)
print("CPU usage after prediction:", cpu_usage_after)
print("Memory usage after prediction:", memory_usage_after)
print("Prediction time:", prediction_time)

# Define the file path where you want to save the output
output_file_path = f_performance_file

# Assuming saved_model is your trained model

# Evaluate the model on validation data
val_loss, val_accuracy = saved_model.evaluate(X_val, y_val)

# Predict classes on validation data
y_val_pred_proba = saved_model.predict(X_val)
y_val_pred_classes = (y_val_pred_proba > 0.5).astype(int)

# Calculate performance metrics
val_precision = precision_score(y_val, y_val_pred_classes)
val_recall = recall_score(y_val, y_val_pred_classes)
val_f1 = f1_score(y_val, y_val_pred_classes)
val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)
val_avg_precision = average_precision_score(y_val, y_val_pred_proba)

# Compute confusion matrix
val_conf_matrix = confusion_matrix(y_val, y_val_pred_classes)

# Print or display validation metrics
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')
print(f'Validation Precision: {val_precision}')
print(f'Validation Recall: {val_recall}')
print(f'Validation F1 Score: {val_f1}')
print(f'Validation ROC AUC Score: {val_roc_auc}')
print(f'Validation Average Precision Score: {val_avg_precision}')
print(f'val_confusion_matrix:  {val_conf_matrix}')

early_epoch=epochs_to_run
last_epoch=epochs_to_run
epstop="None"

# Retrieve the epoch at which training stopped (early stopping)
if hasattr(early_stopping, 'stopped_epoch'):
    early_epoch = early_stopping.stopped_epoch + 1  # Increment by 1 to match epoch indexing
    print(f"Training stopped at epoch: {early_epoch}")
    epstop="early"
else:
    # If early stopping was not triggered, retrieve the total number of epochs completed
    last_epoch = len(history.history['loss'])
    print(f"Total epochs completed: {last_epoch}")
    epstop="last"
    
# Open the file in write mode
with open(output_file_path, 'w') as file:
    # Write performance metrics to the file
    file.write("# Shallow 20 Performance Metrics \n")
    file.write(f"Epochs to run:{epochs_to_run}\n")
    file.write(f"Last epoch {epstop} :{last_epoch}\n")
    file.write(f"Early stop epoch {epstop} :{early_epoch}\n")
    file.write(f"Training_time:{training_time}\n")
    file.write(f"Training Loss: {train_loss}\n")
    file.write(f"Training Accuracy: {train_accuracy}\n")
    file.write(f"Validation Loss: {val_loss}\n")
    file.write(f"Validation Accuracy: {val_accuracy}\n")
    file.write(f"Epoch with Maximum Validation Accuracy: {max_val_acc_epoch}\n")
    file.write(f"Maximum Validation Accuracy: {max_val_acc}\n")
    file.write(f"Epoch with Minimum Validation Loss: {min_val_loss_epoch}\n")
    file.write(f"Minimum Validation Loss: {min_val_loss}\n")
    
    file.write(f"Test Accuracy: {accuracy}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"F1 Score: {f1}\n")
    file.write(f"ROC AUC Score: {roc_auc}\n")
    file.write(f"Average Precision: {avg_precision}\n")
    file.write(f"Confusion Matrix:\n{conf_matrix}\n")
    # Write system information to the file
    
    file.write("\n# System Information\n")
    file.write(f"CPU Usage Before Prediction: {cpu_usage_before}\n")
    file.write(f"Memory Usage Before Prediction: {memory_usage_before}\n")
    file.write(f"CPU Usage After Prediction: {cpu_usage_after}\n")
    file.write(f"Memory Usage After Prediction: {memory_usage_after}\n")
    file.write(f"Prediction Time: {prediction_time} seconds\n")

    # Print or display validation metrics
    file.write(f'Validation Loss: {val_loss}\n')
    file.write(f'Validation Accuracy: {val_accuracy}\n')
    file.write(f'Validation Precision: {val_precision}\n')
    file.write(f'Validation Recall: {val_recall}\n')
    file.write(f'Validation F1 Score: {val_f1}\n')
    file.write(f'Validation ROC AUC Score: {val_roc_auc}\n')
    file.write(f'Validation Average Precision Score: {val_avg_precision}\n')
    file.write(f"Validation Confusion Matrix:\n{val_conf_matrix}\n")

# Create a new figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
# Ensure that epochs are defined with the correct length matching the number of data points
epochs_t=min_val_loss_epoch
#epochs_v=last_epoch #min_val_loss_epoch


# Create a figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].axvline(x=epochs_t, color='g', linestyle='--', label='Min Loss Epoch ' + str(epochs_t))# Vertical line at epoch 60
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# Plot loss
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].axvline(x=epochs_t, color='g', linestyle='--', label='Min Loss Epoch '+str(epochs_t)) # Vertical line at epoch 60
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()

# Save the figure as a JPG file (assuming datapath is defined)
plt.savefig(datapath + slash + xmodel + '_loss_accuracy.png')
plt.close()

print(datapath + slash + xmodel +'_loss_accuracy.png')

# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred)

# Plot ROC curve and Precision-Recall curve as subplots
plt.figure(figsize=(15, 6))

# Plot ROC curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Shallow Model')
plt.legend(loc="lower right")

# Plot Precision-Recall curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (AP = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.tight_layout()

# Save the figure as a JPEG file
plt.savefig(datapath + slash + xmodel +'_roc_pr_curves.png')

# Show the plots
plt.close()
print(datapath + slash + xmodel + '_roc_pr_curves.jpg')

# Print a message indicating that the output was saved
print(f"Performance metrics and system information saved to: {output_file_path}")

