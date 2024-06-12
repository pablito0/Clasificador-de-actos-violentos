import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Configuración inicial
video_dir = r".\Videos"
categories = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
n_classes = len(categories)
frame_size = (90, 90)
seq_length = 16

def load_videos(video_dir, categories, frame_size, seq_length):
    X, y = [], []
    for idx, category in enumerate(categories):
        category_path = os.path.join(video_dir, category)
        for video_name in os.listdir(category_path):
            video_path = os.path.join(category_path, video_name)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = total_frames // seq_length
            frames = [cv2.resize(cv2.VideoCapture(video_path).read()[1], frame_size) for i in range(seq_length) if cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)]
            cap.release()
            if len(frames) == seq_length:
                X.append(frames)
                y.append(idx)
    return np.array(X), np.array(y)

# Definición del modelo
input_shape = (16, 90, 90, 3)
model_input = Input(shape=input_shape)
x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(model_input)
x = BatchNormalization()(x)
x = MaxPooling3D((1, 2, 2), strides=(1, 2, 2))(x)
x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
model_output = Dense(n_classes, activation='softmax')(x)
model = Model(inputs=model_input, outputs=model_model_output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Carga y preparación de datos
X, y = load_videos(video_dir, categories, frame_size, seq_length)
X /= 255.0
y_cat = to_categorical(y, num_classes=n_classes)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Entrenamiento y evaluación
history = model.fit(X_train, y_train, epochs=50, batch_size=1, validation_data=(X_test, y_test))
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test:: test_accuracy:.3f}')
y_pred = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
