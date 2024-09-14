import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

X_train = np.load('C:/Users/91995/Desktop/deepfake_detection/X_train.npy')
y_train = np.load('C:/Users/91995/Desktop/deepfake_detection/y_train.npy')
X_val = np.load('C:/Users/91995/Desktop/deepfake_detection/X_val.npy')
y_val = np.load('C:/Users/91995/Desktop/deepfake_detection/y_val.npy')

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (Real or Fake)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=10,  
    validation_data=(X_val, y_val)
)

model.save('C:/Users/91995/Desktop/deepfake_detection/deepfake_detection_model.keras')

loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation loss: {loss}')
print(f'Validation accuracy: {accuracy}')

