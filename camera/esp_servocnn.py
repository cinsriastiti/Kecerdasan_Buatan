import numpy as np
from keras.preprocessing import image
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import serial 
import time   

try:
    ser = serial.Serial('COM4', 115200, timeout=1) 
    time.sleep(2) 
    print("Serial port COM4 berhasil dibuka.")
except serial.SerialException as e:
    print(f"Error membuka serial port: {e}")
    print("Pastikan ESP32 terhubung dan driver sudah terinstal. Program akan keluar.")
    exit() 

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

print("Memulai pelatihan model CNN...")
cnn.fit(x=training_set, validation_data=test_set, epochs=25)
print("Pelatihan model selesai.")

vid = cv2.VideoCapture(1) 
if not vid.isOpened():
    print("Gagal membuka kamera. Pastikan kamera terhubung dan tidak digunakan oleh aplikasi lain.")
    ser.close()
    exit()


class_labels = sorted(training_set.class_indices.keys(), key=lambda k: training_set.class_indices[k])
print(f"Label kelas terdeteksi: {class_labels}")

threshold = {
    'Apel': 0.75,
    'Pisang': 0.70,
    'Anggur': 0.80
}


print("\nMemulai deteksi objek dari webcam. Tekan 'q' untuk keluar.")
while True:
    r, frame = vid.read()
    if not r:
        print("Gagal membaca frame dari kamera. Mengakhiri program.")
        break

    img_path = 'temp.jpg'
    cv2.imwrite(img_path, frame)

    test_img = image.load_img(img_path, target_size=(64, 64))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img /= 255.0

    prediction = cnn.predict(test_img, verbose=0)
    label_index = np.argmax(prediction)
    predicted_label = class_labels[label_index] 
    confidence = prediction[0][label_index]

    height, width, _ = frame.shape
    box_width, box_height = 200, 200
    start_x = width // 2 - box_width // 2
    start_y = height // 2 - box_height // 2
    end_x = start_x + box_width
    end_y = start_y + box_height
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    current_threshold = threshold.get(predicted_label, 0.8) 

    if confidence >= current_threshold:
        display_text = f'{predicted_label} ({confidence:.2f})'

        try:
            ser.write((predicted_label + "\n").encode())
            print(f"Dikirim ke ESP32: {predicted_label} (Confidence: {confidence:.2f})")
        except Exception as e:
            print(f"Error mengirim data serial: {e}")
            pass 

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
if ser.is_open:
    ser.close() 
print("Program selesai dan serial port ditutup.")
