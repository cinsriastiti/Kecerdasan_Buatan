import numpy as np
from keras.preprocessing import image
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn.fit(x=training_set, validation_data=test_set, epochs=25)

vid = cv2.VideoCapture(1)
class_labels = list(training_set.class_indices.keys()) 

threshold = {
    'Anggur': 0.80,
    'Pisang': 0.70,
    'Apel': 0.75
}

while True:
    r, frame = vid.read()
    if not r:
        break

    img_path = 'temp.jpg'
    cv2.imwrite(img_path, frame)

    test_img = image.load_img(img_path, target_size=(64, 64))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img /= 255.0

    prediction = cnn.predict(test_img, verbose=0)
    label_index = np.argmax(prediction)
    label = class_labels[label_index]
    confidence = prediction[0][label_index]

    height, width, _ = frame.shape
    box_width, box_height = 200, 200
    start_x = width // 2 - box_width // 2
    start_y = height // 2 - box_height // 2
    end_x = start_x + box_width
    end_y = start_y + box_height
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    if confidence >= threshold.get(label, 0.8):
        display_text = f'{label} ({confidence:.2f})'
    else:
        display_text = f'Unknown ({confidence:.2f})'

    cv2.putText(frame, display_text, (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Classification', frame)
    os.remove(img_path)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
