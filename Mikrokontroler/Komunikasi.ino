#include <ESP32Servo.h>

Servo servo1;
Servo servo2;
Servo servo3;

void setup() {
  Serial.begin(115200);
  // Pastikan pin ini adalah pin PWM yang benar di ESP32 Anda!
  // Misalnya, GPIO13, GPIO12, GPIO11 (jika pin 11 valid untuk servo di ESP32)
  servo1.attach(13);
  servo2.attach(12);
  servo3.attach(14); // Double check if pin 11 is a valid PWM pin for servo on your ESP32 board
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n'); // Membaca hingga karakter newline
    input.trim(); // Menghilangkan spasi atau karakter tidak terlihat di awal/akhir

    // Pastikan string "Apel", "Pisang", "Anggur" sama persis
    // dengan yang dikirim dari Python (termasuk kapitalisasi!)
    if (input == "Apel") {
      Serial.println("Menerima Apel"); // Debugging
      servo1.write(30);   // Arah ke tempat apel
      delay(1000);
      servo1.write(90);   // Kembali ke posisi netral
    } else if (input == "Pisang") {
      Serial.println("Menerima Pisang"); // Debugging
      servo2.write(90);   // Arah ke tempat pisang
      delay(1000);
      servo2.write(90);   // Kembali ke posisi netral (posisi awal sudah 90, jadi bisa dihilangkan)
    } else if (input == "Anggur") {
      Serial.println("Menerima Anggur"); // Debugging
      servo3.write(150);  // Arah ke tempat anggur
      delay(1000);
      servo3.write(90);   // Kembali ke posisi netral
    }
  }
}