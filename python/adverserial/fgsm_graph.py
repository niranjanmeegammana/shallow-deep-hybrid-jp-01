import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod

# Argument Parser for Command Line Inputs
parser = argparse.ArgumentParser(description="FGSM Adversarial Attack on Neural Network")
parser.add_argument("--eps", type=float, default=0.2, help="Epsilon value for FGSM attack (perturbation strength)")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
args = parser.parse_args()

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, 
                           n_informative=2, n_redundant=0, random_state=42)

# Build a simple neural network
model = Sequential([
    Input(shape=(2,)),
    Dense(10, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"])

# Train the model (wrapped in tf.function for ART compatibility)
@tf.function
def train_model():
    model.fit(X, y, epochs=args.epochs, verbose=1, batch_size=args.batch_size)

train_model()

# Wrap the model with ART's KerasClassifier
classifier = KerasClassifier(model=model, clip_values=(X.min(), X.max()))

# Generate adversarial samples using FGSM
fgsm_attack = FastGradientMethod(estimator=classifier, eps=args.eps)
X_adv = fgsm_attack.generate(X)

# Select a sample to visualize
idx = 0
original_sample = X[idx]
perturbed_sample = X_adv[idx]

# Plot original vs perturbed sample
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.3, label="Original Data")
plt.scatter(original_sample[0], original_sample[1], color="blue", label="Original Sample", edgecolors="black", s=100)
plt.scatter(perturbed_sample[0], perturbed_sample[1], color="red", label="Perturbed Sample", edgecolors="black", s=100)
plt.legend()
plt.title(f"FGSM Attack (Eps={args.eps}): Original vs Perturbed Sample")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid()
plt.savefig("fgsm_attack_result.png")  # Save the figure
plt.show()

# Print original and perturbed values
print("\nOriginal Sample:", original_sample)
print("Perturbed Sample:", perturbed_sample)
