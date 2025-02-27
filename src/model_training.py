import numpy as np
from mlp_model import FFSNNetwork
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load training and test data
X_train = np.load('./processed_data/xtrain.npy')
y_train = np.load('./processed_data/ytrain.npy')
X_test = np.load('./processed_data/xtest.npy')
y_test = np.load('./processed_data/ytest.npy')

# Initialize and train the model
mlp = FFSNNetwork(n_inputs=X_train.shape[1], hidden_sizes=[15])  # Initialize the network with 15 hidden neurons
mlp.fit(X_train, y_train, epochs=3000, learning_rate=0.01, display_loss=True)

# Save the weights and biases
np.save('./output/weights.npy', mlp.W)
np.save('./output/biases.npy', mlp.B)

# Prediction and evaluation
y_pred = mlp.predict(X_test)  # Predicted probabilities
y_pred_classes = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Test Accuracy: {accuracy}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Plot and save the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('./output/confusion_matrix.png')
plt.show()

print("Model trained, evaluated, and results saved.")
