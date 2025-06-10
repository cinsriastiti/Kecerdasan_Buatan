import numpy as np
from keras.preprocessing import image
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import serial # Import modul serial
import time   # Import modul time untuk delay

# --- Bagian Komunikasi Serial (BARU) ---
try:
    # Ganti 'COM4' dengan port serial ESP32 Anda
    # Anda bisa cek di Device Manager (Windows) atau /dev/ttyUSBX (Linux/macOS)
    ser = serial.Serial('COM4', 115200, timeout=1) # Tambahkan timeout agar tidak macet
    time.sleep(2) # Beri waktu ESP32 untuk inisialisasi
    print("Serial port COM4 berhasil dibuka.")
except serial.SerialException as e:
    print(f"Error membuka serial port: {e}")
    print("Pastikan ESP32 terhubung dan driver sudah terinstal. Program akan keluar.")
    exit() # Keluar dari program jika serial tidak bisa dibuka

# --- Data preprocessing ---
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

# --- CNN model ---
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax')) # Pastikan units=3 sesuai jumlah kelas

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Training (Pastikan 'train' dan 'test' folder ada di lokasi yang benar) ---
print("Memulai pelatihan model CNN...")
cnn.fit(x=training_set, validation_data=test_set, epochs=25)
print("Pelatihan model selesai.")

# --- Webcam prediction ---
vid = cv2.VideoCapture(1) # Ganti 0 jika kamera utama PC, 1 jika kamera eksternal
if not vid.isOpened():
    print("Gagal membuka kamera. Pastikan kamera terhubung dan tidak digunakan oleh aplikasi lain.")
    ser.close()
    exit()

# Mendapatkan nama kelas dari training_set.class_indices
# Ini penting agar urutan dan nama label konsisten dengan folder data Anda
class_labels = sorted(training_set.class_indices.keys(), key=lambda k: training_set.class_indices[k])
print(f"Label kelas terdeteksi: {class_labels}")

# Thresholds per class (SESUAIKAN DENGAN NAMA FOLDER DATASET ANDA!)
# Contoh: Jika nama folder Anda 'Apple', 'Banana', 'Grape'
# maka ubah 'Apel' menjadi 'Apple', 'Pisang' menjadi 'Banana', 'Anggur' menjadi 'Grape'
threshold = {
    'Apel': 0.75,
    'Pisang': 0.70,
    'Anggur': 0.80
}

# Pastikan label di 'threshold' dictionary sesuai dengan nama folder di dataset Anda
# Contoh: Jika folder Anda 'apple', 'banana', 'grape', maka ubah:
# threshold = {
#     'apple': 0.75,
#     'banana': 0.70,
#     'grape': 0.80
# }

print("\nMemulai deteksi objek dari webcam. Tekan 'q' untuk keluar.")
while True:
    r, frame = vid.read()
    if not r:
        print("Gagal membaca frame dari kamera. Mengakhiri program.")
        break

    # Simpan frame sementara untuk prediksi
    img_path = 'temp.jpg'
    cv2.imwrite(img_path, frame)

    # Pra-pemrosesan gambar untuk model CNN
    test_img = image.load_img(img_path, target_size=(64, 64))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img /= 255.0

    # Lakukan prediksi
    prediction = cnn.predict(test_img, verbose=0)
    label_index = np.argmax(prediction)
    predicted_label = class_labels[label_index] # Nama label yang terdeteksi
    confidence = prediction[0][label_index]

    # Bounding box setup
    height, width, _ = frame.shape
    box_width, box_height = 200, 200
    start_x = width // 2 - box_width // 2
    start_y = height // 2 - box_height // 2
    end_x = start_x + box_width
    end_y = start_y + box_height
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Ambil threshold spesifik untuk label yang diprediksi, jika ada
    # Jika tidak ada, gunakan threshold default 0.8 atau sesuaikan
    current_threshold = threshold.get(predicted_label, 0.8) # Mengambil threshold dari dictionary

    # Aplikasikan keputusan berdasarkan threshold
    if confidence >= current_threshold:
        display_text = f'{predicted_label} ({confidence:.2f})'
        # --- Kirim perintah ke ESP32 (BARU) ---
        try:
            # Pastikan predicted_label ini sama persis dengan yang di ESP32 (misal: "Apel", "Pisang", "Anggur")
            # Jika dataset Anda menggunakan 'apple', 'banana', 'grape' (huruf kecil),
            # Anda perlu mengubahnya menjadi 'Apel', 'Pisang', 'Anggur' (kapitalisasi seperti di Arduino)
            # Misalnya:
            # if predicted_label.lower() == 'apple':
            #     label_to_send = 'Apel'
            # elif predicted_label.lower() == 'banana':
            #     label_to_send = 'Pisang'
            # elif predicted_label.lower() == 'grape':
            #     label_to_send = 'Anggur'
            # else:
            #     label_to_send = "Unknown" # Atau biarkan kosong

            # Saat ini, saya berasumsi predicted_label sudah sesuai dengan kapitalisasi di Arduino
            # Jika tidak, Anda perlu menambahkan logika konversi di atas
            ser.write((predicted_label + "\n").encode())
            print(f"Dikirim ke ESP32: {predicted_label} (Confidence: {confidence:.2f})")
        except Exception as e:
            print(f"Error mengirim data serial: {e}")
            pass # Lanjutkan program meski ada error serial

    else:
        display_text = f'Unknown ({confidence:.2f})'
        # Opsional: Kirim sinyal "Unknown" ke ESP32 jika tidak ada yang terdeteksi dengan yakin
        # ser.write(b"Unknown\n") # Kirim string byte "Unknown" diikuti newline

    # Tampilkan label di layar
    cv2.putText(frame, display_text, (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow('Classification', frame)
    os.remove(img_path) # Hapus file gambar sementara

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Pembersihan setelah loop berakhir ---
vid.release()
cv2.destroyAllWindows()
if ser.is_open:
    ser.close() # Pastikan serial port ditutup
print("Program selesai dan serial port ditutup.")