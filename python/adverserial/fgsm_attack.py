import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier


xmodel= "concat_40"
model_path = f"d:\\miniconda\\UNSW-NB15\\testing\\JP1_HB_V3\\hy_models\\shadeep_hyb_{xmodel}_model.keras"
# Verify Model Path Exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

print(f"Loading Model: {model_path}")
model = tf.keras.models.load_model(model_path)

# 📌 Load Data
df_train = pd.read_csv("d:\\miniconda\\UNSW-NB15\\testing\\data\\unsw-nb15_training_90_5_5b_.csv")
df_test = pd.read_csv("d:\\miniconda\\UNSW-NB15\\testing\\data\\unsw-nb15_testing_90_5_5b_.csv")

# 📌 Define Features
features = ['id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 
            'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
            'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 
            'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

# 📌 Prepare Data
X_train = df_train[features].values
y_train = df_train['label'].values
X_test = df_test[features].values
y_test = df_test['label'].values

# 📌 Normalize Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 📌 Convert Labels to Float Format
y_train = y_train.astype(np.float32).reshape(-1, 1)
y_test = y_test.astype(np.float32).reshape(-1, 1)

# 📌 Convert Model to ART Classifier
classifier = TensorFlowV2Classifier(
    model=model,
    nb_classes=2,
    input_shape=(X_train.shape[1],),
    clip_values=(X_train.min().astype(np.float32), X_train.max().astype(np.float32)),
    loss_object=tf.keras.losses.BinaryCrossentropy(),
)

# 📌 Generate FGSM Adversarial Samples
fgsm_attack = FastGradientMethod(estimator=classifier, eps=0.02)  # Low attack strength
X_train_adv = fgsm_attack.generate(X_train)
X_test_adv = fgsm_attack.generate(X_test)

# 📌 Combine Clean & Adversarial Data for PCA
X_combined = np.vstack([X_test, X_test_adv])  # Merge clean and adversarial test data
y_combined = np.hstack([y_test.flatten(), y_test.flatten()])  # Keep labels the same

# 📌 Apply PCA for Dimensionality Reduction (Reduce to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined)

# 📌 Plot PCA Results
plt.figure(figsize=(8, 6), dpi=300)
sns.scatterplot(x=X_pca[:len(X_test), 0], y=X_pca[:len(X_test), 1], label="Clean Samples", alpha=0.5)
sns.scatterplot(x=X_pca[len(X_test):, 0], y=X_pca[len(X_test):, 1], label="Adversarial Samples", alpha=0.5)
plt.title(f"PCA of Clean vs. FGSM Adversarial Samples ({xmodel})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.savefig(f"PCA_Clean_vs_Adversarial_{xmodel}.png", dpi=300, bbox_inches="tight")
plt.show()

# 📌 Evaluate Model Performance on Clean & Adversarial Data
clean_acc = np.mean((model.predict(X_test) > 0.5).astype(int) == y_test) * 100
adv_acc = np.mean((model.predict(X_test_adv) > 0.5).astype(int) == y_test) * 100

print(f"\n📌 Model: {xmodel}")
print(f"✅ Accuracy on Clean Test Data: {clean_acc:.2f}%")
print(f"⚠️ Accuracy on Adversarial Test Data: {adv_acc:.2f}%")


