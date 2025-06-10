#include <ESP32Servo.h>

Servo servo1;
Servo servo2;
Servo servo3;

void setup() {
  Serial.begin(115200);
  servo1.attach(13);
  servo2.attach(12);
  servo3.attach(14); 
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n'); 
    input.trim(); 

    if (input == "Apel") {
      Serial.println("Menerima Apel");
      servo1.write(30);  
      delay(1000);
      servo1.write(90); 
    } else if (input == "Pisang") {
      Serial.println("Menerima Pisang"); 
      servo2.write(90);
      delay(1000);
      servo2.write(90); 
    } else if (input == "Anggur") {
      Serial.println("Menerima Anggur");
      servo3.write(150);
      delay(1000);
      servo3.write(90); 
    }
  }
}
