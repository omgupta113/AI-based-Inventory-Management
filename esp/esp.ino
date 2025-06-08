#include <WiFi.h>
#include <HTTPClient.h>

// WiFi credentials
const char* ssid = "om";
const char* password = "12344321";

// API endpoint
const char* apiEndpoint = "http://omlaptop.local:8000/capture";

// L298N motor driver pins for first motor (main conveyor)
const int ENA = 13; // PWM pin for motor speed control
const int IN1 = 12; // Motor direction control 1
const int IN2 = 14; // Motor direction control 2

// L298N motor driver pins for second motor (secondary conveyor)
const int ENB = 15; // PWM pin for second motor speed control
const int IN3 = 2;  // Second motor direction control 1
const int IN4 = 4;  // Second motor direction control 2

// IR Sensor pin
const int IR_SENSOR = 27; // IR sensor pin

// Motor speed (0-255)
const int MOTOR_SPEED = 135;
const int MOTOR2_SPEED = 155; // Speed for the second motor (can be adjusted if needed)

// Time constants
const unsigned long STOP_DURATION = 2000; // 2 seconds in milliseconds

// Motor state variables
bool motorsRunning = true;  // Initialize as running by default
unsigned long startTime = 0;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Configure motor pins for first motor
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  
  // Configure motor pins for second motor
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  
  // Configure IR sensor pin
  pinMode(IR_SENSOR, INPUT);
  
  // Connect to WiFi
  connectToWiFi();
  
  // Wait a moment after WiFi connection before checking sensor
  delay(1000);
  
  // Start motors by default
  startMotors();
  
  // Initial motor state check based on IR sensor
  checkAndUpdateMotors();
}

void loop() {
  checkAndUpdateMotors();
}

void checkAndUpdateMotors() {
  // Read current IR sensor state
  int irState = digitalRead(IR_SENSOR);
  
  // IR HIGH means object detected - motors should be running
  if (irState == HIGH && !motorsRunning) {
    startMotors();
    motorsRunning = true;
    Serial.println("Object detected! Motors started.");
  } 
  // IR LOW means no object - motors should be stopped
  // Only stop motors if they're currently running
  else if (irState == LOW && motorsRunning) {
    delay(600);
    stopMotors();
    delay(3000);
    callCaptureAPI();
    motorsRunning = false;
    startTime = millis();
    Serial.println("No object detected. Motors stopped.");
    delay(1500);
  }
  delay(10);
  
  // If motors are stopped and waiting period has elapsed, check if we should restart
  // if (!motorsRunning && (millis() - startTime >= STOP_DURATION)) {
  //   // Only restart if object is detected
  //   if (digitalRead(IR_SENSOR) == HIGH) {
  //     startMotors();
  //     motorsRunning = true;
  //     Serial.println("Wait time elapsed. Object detected. Restarting motors.");
  //   }
  // }
}

void startMotors() {
  // Start first motor (main conveyor)
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, MOTOR_SPEED);
  
  // Start second motor (secondary conveyor)
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, MOTOR2_SPEED);
  
  Serial.println("Both motors started");
}

void stopMotors() {
  // Stop first motor
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);
  
  // Stop second motor
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, 0);
  
  Serial.println("Both motors stopped");
}

void connectToWiFi() {
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.print("Connected to WiFi, IP address: ");
  Serial.println(WiFi.localIP());
}

void callCaptureAPI() {
  // Check WiFi connection
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    Serial.print("Calling API: ");
    Serial.println(apiEndpoint);
    
    // Begin HTTP connection
    http.begin(apiEndpoint);
    
    // Send GET request and get response code
    int httpResponseCode = http.GET();
    
    if (httpResponseCode > 0) {
      Serial.print("HTTP Response code: ");
      Serial.println(httpResponseCode);
      
      // Optional: Get response payload
      String payload = http.getString();
      Serial.println(payload);
    } else {
      Serial.print("Error code: ");
      Serial.println(httpResponseCode);
    }
    
    // Free resources
    http.end();
  } else {
    Serial.println("WiFi Disconnected. Cannot call API.");
  }
}