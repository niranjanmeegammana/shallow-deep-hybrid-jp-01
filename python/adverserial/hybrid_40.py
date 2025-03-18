# without adverserial training
import psutil
import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
)

current_folder = "d:\\miniconda\\UNSW-NB15\\testing"
slash = "\\"
split = "_90_5_5b_"

datapath = current_folder + slash + "data" + slash
results_path = current_folder + slash + "results" + slash
in_modelpath = current_folder + slash + "models" + slash
out_modelpath = current_folder + slash + "JP1_HB_V3\\hy_models" + slash

f_train = datapath + "unsw-nb15_training" + split + ".csv"
f_test = datapath + "unsw-nb15_testing" + split + ".csv"

# -------------------------
# Load Data
df_train = pd.read_csv(f_train)
df_test = pd.read_csv(f_test)

# Define Features
features = [
    "id","dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports"
]

# Separate Features & Labels
X_train = df_train[features].values
y_train = df_train["label"].values
X_test = df_test[features].values
y_test = df_test["label"].values

# Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# Model Name & Paths
#models = ["concat_40", "maximum_40",  "wgtav_40"]

# Function to Test Model Performance
def modeltest():
    global model, X_test, y_test

    print("\nModel Performance Evaluation:")

    # Measure CPU and Memory Usage Before Prediction
    cpu_usage_before = psutil.cpu_percent(interval=1)  # Allow time to collect accurate data
    memory_usage_before = psutil.Process(os.getpid()).memory_info().rss  # Per-process memory usage

    # Record Prediction Time
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # Measure CPU and Memory Usage After Prediction
    cpu_usage_after = psutil.cpu_percent(interval=1)
    memory_usage_after = psutil.Process(os.getpid()).memory_info().rss

    # Convert Probabilities to Binary Predictions (0 or 1)
    y_pred_classes = (y_pred > 0.5).astype(int)

    # Compute Performance Metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes)
    recall = recall_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes)
    roc_auc = roc_auc_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_pred)

    # Compute Confusion Matrix
    conf_matrix = tf.math.confusion_matrix(y_test, y_pred_classes).numpy()

    # Print Results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Confusion Matrix:\n {conf_matrix}")
    print(f"CPU Usage Before Prediction: {cpu_usage_before}%")
    print(f"Memory Usage Before Prediction: {memory_usage_before / (1024**2):.2f} MB")
    print(f"CPU Usage After Prediction: {cpu_usage_after}%")
    print(f"Memory Usage After Prediction: {memory_usage_after / (1024**2):.2f} MB")
    print(f"Prediction Time: {prediction_time:.4f} seconds")
    return accuracy
results = []  # Initialize an empty list
xmodel="concat_40"
xmodel="maximum_40"
xmodel= "wgtav_40"
model_path = out_modelpath + "shadeep_hyb_"+ xmodel + "_model.keras"
# Load Model
print(model_path)
model = tf.keras.models.load_model(model_path)
#model.summary()
# Run Model Test
clean_accuracy=modeltest()
# Convert labels to categorical format
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# Convert to TensorFlow tensors
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

# -------------------------
# Convert Model for Adversarial Testing
classifier = TensorFlowV2Classifier(
    model=model,
    nb_classes=2,
    input_shape=(X_train.shape[1],),
    clip_values=(X_train.min().astype(np.float32), X_train.max().astype(np.float32)),
    loss_object=tf.keras.losses.CategoricalCrossentropy(),
)

# -------------------------
# Generate Adversarial Samples using FGSM
fgsm_attack = FastGradientMethod(estimator=classifier, eps=0.1)
X_adv = fgsm_attack.generate(X_test_tensor.numpy())

# Convert back to TensorFlow tensor
X_adv_tensor = tf.convert_to_tensor(X_adv, dtype=tf.float32)

# -------------------------
# Evaluate Model on Clean vs. Adversarial Data
#clean_acc = np.mean(np.argmax(model.predict(X_test), axis=1) == np.argmax(y_test, axis=1))
adv_acc = np.mean(np.argmax(model.predict(X_adv_tensor), axis=1) == np.argmax(y_test_tensor, axis=1))

clean_accuracy *= 100
adv_acc *= 100
r=(xmodel, clean_accuracy, adv_acc)
results.append(r) 
print(xmodel)
print(f"nAccuracy on Clean Test Data: {clean_accuracy:.2f}%")
print(f"Accuracy on Adversarial Test Data: {adv_acc:.2f}%")
'''
# -------------------------
# Visualize Accuracy Degradation
plt.figure(figsize=(6, 4), dpi=300)
sns.barplot(x=["Clean Data", "Adversarial Data"], y=[clean_acc, adv_acc], palette=["cyan", "pink"])

plt.ylabel("Accuracy (%)", fontsize=14, fontfamily="serif")
plt.title(f"Impact of FGSM Attack on {xmodel}", fontsize=14, fontfamily="serif")
plt.xticks(fontsize=14, fontfamily="serif")
plt.yticks(fontsize=12, fontfamily="serif")

plt.savefig("FGSM_Adversarial_Impact.png", dpi=300, bbox_inches="tight")
plt.show()
'''

    
# Print table header
print("| Model | Clean Accuracy | Adversarial Accuracy |")
print("|-------|---------------|----------------------|")

# Iterate through results and print each row
for r in results:
    xmodel, clean_acc, adv_acc = r  # Correct variable unpacking
    print(f"| {xmodel} | {clean_acc:.2f}% | {adv_acc:.2f}% |")
