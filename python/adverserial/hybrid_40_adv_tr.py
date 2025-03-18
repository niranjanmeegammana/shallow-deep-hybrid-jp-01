import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.callbacks import EarlyStopping
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier
from sklearn.preprocessing import StandardScaler

# Load Model
#xmodel = "concat_40"
#xmodel="maximum_40"
xmodel= "wgtav_40"
model_path = f"d:\\miniconda\\UNSW-NB15\\testing\\JP1_HB_V3\\hy_models\\shadeep_hyb_{xmodel}_model.keras"
print(model_path)
model = tf.keras.models.load_model(model_path)

#  Load Data
df_train = pd.read_csv("d:\\miniconda\\UNSW-NB15\\testing\\data\\unsw-nb15_training_90_5_5b_.csv")
df_test = pd.read_csv("d:\\miniconda\\UNSW-NB15\\testing\\data\\unsw-nb15_testing_90_5_5b_.csv")

# Define Features
features = ['id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 
            'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
            'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 
            'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

# Prepare Data
X_train = df_train[features].values
y_train = df_train['label'].values
X_test = df_test[features].values
y_test = df_test['label'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert Labels to Binary Format (for `binary_crossentropy`)
y_train = y_train.astype(np.float32).reshape(-1, 1)  # Convert to shape (None, 1)
y_test = y_test.astype(np.float32).reshape(-1, 1)    # Convert to shape (None, 1)

# Convert Model to ART Classifier
classifier = TensorFlowV2Classifier(
    model=model,
    nb_classes=2,
    input_shape=(X_train.shape[1],),
    clip_values=(X_train.min().astype(np.float32), X_train.max().astype(np.float32)),
    loss_object=tf.keras.losses.BinaryCrossentropy(),
)

# Generate FGSM Adversarial Samples
fgsm_attack = FastGradientMethod(estimator=classifier, eps=0.02)  # Start with a low attack strength
X_train_adv = fgsm_attack.generate(X_train)
X_test_adv = fgsm_attack.generate(X_test)

#Combine Clean & Adversarial Data
X_train_combined = np.vstack([X_train, X_train_adv])
y_train_combined = np.vstack([y_train, y_train])  # Labels remain the same

# Compile Model with Early Stopping Callback
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="binary_crossentropy",  # ✅ Matches model output (None,1)
              metrics=["accuracy"])

# Train Model with Early Stopping
history = model.fit(X_train_combined, y_train_combined, 
                    epochs=1,  # Start with a high limit
                    batch_size=64, 
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])
                    
#Identify the Best Epoch
best_epoch = len(history.history["val_loss"]) - 5  # Since `patience=5`, best epoch is when loss last improved

print(f"\nBest Epoch to Stop Training: {best_epoch}")
'''
# ✅ Plot Training vs. Validation Loss
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2, linestyle="dashed")
plt.axvline(best_epoch - 1, color="red", linestyle="dotted", label=f"Best Epoch ({best_epoch})")

# Formatting
plt.xlabel("Epochs", fontsize=14, fontfamily="serif")
plt.ylabel("Loss", fontsize=14, fontfamily="serif")
plt.title("Training vs Validation Loss", fontsize=14, fontfamily="serif")
plt.legend(fontsize=12)
plt.xticks(fontsize=12, fontfamily="serif")
plt.yticks(fontsize=12, fontfamily="serif")

# Save and Show Plot
plt.savefig("Optimal_Training_Epochs.png", dpi=300, bbox_inches="tight")
plt.show()
'''

# Save the Adversarially Trained Model
model.save(f"{xmodel}_adversarial_trained.keras")

# Evaluate Model Performance After Adversarial Training
clean_acc = np.mean((model.predict(X_test) > 0.5).astype(int) == y_test)
adv_acc = np.mean((model.predict(X_test_adv) > 0.5).astype(int) == y_test)

# Convert Accuracy to Percentage
clean_acc *= 100
adv_acc *= 100
print(xmodel)
print(f"Accuracy on Clean Test Data: {clean_acc:.2f}%")
print(f"Accuracy on Adversarial Test Data: {adv_acc:.2f}%")

# Plot Training History
plt.figure(figsize=(12, 5), dpi=300)

# 📌 Plot Validation Loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.axvline(best_epoch, color='r', linestyle='dashed', label="Best Epoch")
plt.xlabel("Epochs", fontsize=14, fontfamily="serif")
plt.ylabel("Loss", fontsize=14, fontfamily="serif")
plt.title("Training vs Validation Loss", fontsize=14, fontfamily="serif")
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.axvline(best_epoch, color='r', linestyle='dashed', label="Best Epoch")
plt.xlabel("Epochs", fontsize=14, fontfamily="serif")
plt.ylabel("Accuracy", fontsize=14, fontfamily="serif")
plt.title("Training vs Validation Accuracy", fontsize=14, fontfamily="serif")
plt.legend()

# Save and Show Plots
plt.tight_layout()
fig=xmodel+"_Training_Validation_Loss_Accuracy.png"
plt.savefig(fig, dpi=300, bbox_inches="tight")
plt.show()
