import psutil
import time
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import tensorflow as tf
import os
import pandas as pd


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l1, l2
#from kerastuner import HyperParameters
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import StandardScaler

current_folder = os.getcwd()
print(current_folder)
slash="\\"
split="_90_5_5b_"
datapath=current_folder +slash + "data" + slash
modelpath=current_folder + slash + "14002" + slash
f_train=datapath+'unsw-nb15_training' + split +'.csv'
f_test=datapath + 'unsw-nb15_testing' + split+ '.csv'
f_performance_shallow20=datapath + 'shallow20.txt'

shallow_model_path_20=modelpath+ 'shallow_model_ANN5_hp.h5'
deep_model_path_20=modelpath+ 'deep_model_ANN5_hp.h5'

print(f_test)
print(f_train)
print(shallow_model_path_20)

#print(deep_model_path_20)

# Define the features
features = ['ct_state_ttl', 'sload', 'rate', 'sttl', 'smean', 'dload', 'sbytes', 'ct_srv_dst', 'ct_dst_src_ltm', 'dbytes', 'ackdat', 'dttl', 'ct_dst_sport_ltm', 'dmean','ct_srv_src', 'dinpkt', 'tcprtt', 'dur', 'synack', 'sinpkt']

df_train = pd.read_csv(f_train)
df_test = pd.read_csv(f_test)
print(df_test.head(5))

# Separate features and labels for training set
X_train = df_train[features].values # assuming 'label' is the column containing labels
y_train = df_train['label'].values

# Separate features and labels for testing set
X_test = df_test[features].values  # assuming 'label' is the column containing labels
y_test = df_test['label'].values


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Define a learning rate scheduler function
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def build_shallow_ann_model(hp):
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




# Define the filepath to save the model
checkpoint_filepath = shallow_model_path_20

# Define a ModelCheckpoint callback to save the model with the highest validation accuracy
checkpoint = ModelCheckpoint(checkpoint_filepath,
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max',
                             verbose=1)
# Build the shallow ANN model using the best hyperparameters
shallow_ann_model = build_shallow_ann_model(best_hyperparameters)
max_val_acc_epoch=37

# Train the model with early stopping and ModelCheckpoint callback
history = shallow_ann_model.fit(X_train, y_train,
                          epochs=max_val_acc_epoch,
                          validation_split=0.2,
                          callbacks=[checkpoint])


# Save the shallow ANN model to shallow_model_path_20
shallow_ann_model.save(shallow_model_path_20)

# Print model summary
shallow_ann_model.summary()


# Load the saved model
saved_model = load_model(shallow_model_path_20)

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
confusion_matrix = tf.math.confusion_matrix(y_test, y_pred_classes).numpy()

# Print performance metrics
print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Average Precision:", avg_precision)
print("Confusion Matrix:\n", confusion_matrix)

# Print CPU and memory usage, and prediction time
print("CPU usage before prediction:", cpu_usage_before)
print("Memory usage before prediction:", memory_usage_before)
print("CPU usage after prediction:", cpu_usage_after)
print("Memory usage after prediction:", memory_usage_after)
print("Prediction time:", prediction_time)

# Define the file path where you want to save the output
output_file_path = f_performance_shallow20

# Open the file in write mode
with open(output_file_path, 'w') as file:
    # Write performance metrics to the file
    file.write("# Shallow 20 Performance Metrics \n")
    file.write(f"Test Accuracy: {accuracy}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"F1 Score: {f1}\n")
    file.write(f"ROC AUC Score: {roc_auc}\n")
    file.write(f"Average Precision: {avg_precision}\n")
    file.write(f"Confusion Matrix:\n{confusion_matrix}\n")

    # Write system information to the file
    file.write("\n# System Information\n")
    file.write(f"CPU usage before prediction: {cpu_usage_before}\n")
    file.write(f"Memory usage before prediction: {memory_usage_before}\n")
    file.write(f"CPU usage after prediction: {cpu_usage_after}\n")
    file.write(f"Memory usage after prediction: {memory_usage_after}\n")
    file.write(f"Prediction time: {prediction_time} seconds\n")

# Print a message indicating that the output was saved
print(f"Performance metrics and system information saved to: {output_file_path}")

