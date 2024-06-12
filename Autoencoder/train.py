from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import load_model
import numpy as np

X_train = np.load('processed_images.npy')
total_frames = X_train.shape[2]
total_frames -= total_frames % 10

X_train = X_train[:, :, :total_frames]
X_train = X_train.reshape(-1, 227, 227, 10)
X_train = np.expand_dims(X_train, axis=4)
Y_train = X_train.copy()

num_epochs = 20
batch_size = 1

if __name__ == "__main__":
    model = load_model()

    checkpoint = ModelCheckpoint("Autoencoder.h5", monitor="mean_squared_error", save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, callbacks=[checkpoint, early_stopping])
