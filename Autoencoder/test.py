from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix

def mean_squared_loss(x1, x2):
    diff = x1 - x2
    a, b, c, d, e = diff.shape
    n_samples = a * b * c * d * e
    sq_diff = diff ** 2
    Sum = sq_diff.sum()
    dist = np.sqrt(Sum)
    mean_dist = dist / n_samples
    return mean_dist

threshold = 0.1

model = load_model(r"autoencoder.h5")
X_test = np.load('test.npy')
frames = X_test.shape[2]
frames -= frames % 10

X_test = X_test[:, :, :frames]
X_test = X_test.reshape(-1, 227, 227, 10)
X_test = np.expand_dims(X_test, axis=4)

anomalies_detected = []
true_labels = []

for number, bunch in enumerate(X_test):
    n_bunch = np.expand_dims(bunch, axis=0)
    reconstructed_bunch = model.predict(n_bunch)
    loss = mean_squared_loss(n_bunch, reconstructed_bunch)
    
    if loss > threshold:
        anomalies_detected.append(1)
    else:
        anomalies_detected.append(0)

    true_label = determine_true_label(number)  # Placeholder function
    true_labels.append(true_label)

conf_matrix = confusion_matrix(true_labels, anomalies_detected)
print("Confusion Matrix:")
print(conf_matrix)
