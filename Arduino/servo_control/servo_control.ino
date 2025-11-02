/*
 * Arduino Servo Control for Prosthetic Hand
 * Controls 5 servomotors (SPT5632) using PCA9685 PWM driver
 * Receives commands from Raspberry Pi via serial communication
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Create PCA9685 object
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo configuration
#define NUM_SERVOS 5
#define SERVO_FREQ 50  // 50Hz for servos

// Servo channels on PCA9685 (0-15)
const int servoChannels[NUM_SERVOS] = {0, 1, 2, 3, 4};  // Thumb, Index, Middle, Ring, Pinky

// Servo pulse width limits (adjust based on your SPT5632 specs)
#define SERVOMIN  150   // Minimum pulse width (0 degrees)
#define SERVOMAX  600   // Maximum pulse width (180 degrees)

// Current servo positions
int currentPositions[NUM_SERVOS] = {90, 90, 90, 90, 90};  // Start at middle position
int targetPositions[NUM_SERVOS] = {90, 90, 90, 90, 90};

// Smooth movement parameters
const int moveSpeed = 2;      // Degrees per step
const int moveDelay = 20;     // Milliseconds between steps

// Serial communication
String inputString = "";
boolean stringComplete = false;

void setup() {
  Serial.begin(9600);
  Serial.println("Prosthetic Hand Controller Starting...");
  
  // Initialize PCA9685
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);  // Set oscillator frequency
  pwm.setPWMFreq(SERVO_FREQ);
  
  // Initialize servos to middle position
  for (int i = 0; i < NUM_SERVOS; i++) {
    setServoAngle(i, currentPositions[i]);
  }
  
  Serial.println("Servos initialized. Ready for commands.");
  Serial.println("Command format: S,angle1,angle2,angle3,angle4,angle5");
  
  delay(1000);  // Give servos time to reach initial position
}

void loop() {
  // Check for serial commands
  if (stringComplete) {
    processCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
  
  // Smooth servo movement
  updateServoPositions();
  
  delay(moveDelay);
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}

void processCommand(String command) {
  if (command.startsWith("S,")) {
    // Parse servo command: S,angle1,angle2,angle3,angle4,angle5
    String angles = command.substring(2);  // Remove "S," prefix
    
    int angleArray[NUM_SERVOS];
    int angleIndex = 0;
    
    // Parse comma-separated angles
    int startIndex = 0;
    for (int i = 0; i <= angles.length() && angleIndex < NUM_SERVOS; i++) {
      if (i == angles.length() || angles.charAt(i) == ',') {
        String angleStr = angles.substring(startIndex, i);
        angleArray[angleIndex] = angleStr.toInt();
        
        // Constrain angle to valid range
        angleArray[angleIndex] = constrain(angleArray[angleIndex], 0, 180);
        
        angleIndex++;
        startIndex = i + 1;
      }
    }
    
    // Set target positions if we got all 5 angles
    if (angleIndex == NUM_SERVOS) {
      for (int i = 0; i < NUM_SERVOS; i++) {
        targetPositions[i] = angleArray[i];
      }
      
      Serial.print("Target positions set: ");
      for (int i = 0; i < NUM_SERVOS; i++) {
        Serial.print(targetPositions[i]);
        if (i < NUM_SERVOS - 1) Serial.print(",");
      }
      Serial.println();
    } else {
      Serial.println("Error: Invalid command format");
    }
  } else if (command.startsWith("STOP")) {
    // Emergency stop - hold current positions
    for (int i = 0; i < NUM_SERVOS; i++) {
      targetPositions[i] = currentPositions[i];
    }
    Serial.println("Emergency stop - holding current positions");
  } else if (command.startsWith("HOME")) {
    // Return to home position (middle)
    for (int i = 0; i < NUM_SERVOS; i++) {
      targetPositions[i] = 90;
    }
    Serial.println("Returning to home position");
  } else if (command.startsWith("STATUS")) {
    // Report current status
    Serial.print("Current positions: ");
    for (int i = 0; i < NUM_SERVOS; i++) {
      Serial.print(currentPositions[i]);
      if (i < NUM_SERVOS - 1) Serial.print(",");
    }
    Serial.println();
  } else {
    Serial.println("Unknown command: " + command);
  }
}

void updateServoPositions() {
  boolean anyMoving = false;
  
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (currentPositions[i] != targetPositions[i]) {
      anyMoving = true;
      
      // Move towards target position
      if (currentPositions[i] < targetPositions[i]) {
        currentPositions[i] = min(currentPositions[i] + moveSpeed, targetPositions[i]);
      } else {
        currentPositions[i] = max(currentPositions[i] - moveSpeed, targetPositions[i]);
      }
      
      // Update servo position
      setServoAngle(i, currentPositions[i]);
    }
  }
  
  // Optional: Print status when movement is complete
  static boolean wasMoving = false;
  if (wasMoving && !anyMoving) {
    Serial.println("Movement complete");
  }
  wasMoving = anyMoving;
}

void setServoAngle(int servoIndex, int angle) {
  if (servoIndex < 0 || servoIndex >= NUM_SERVOS) {
    return;  // Invalid servo index
  }
  
  // Convert angle (0-180) to pulse width
  int pulseWidth = map(angle, 0, 180, SERVOMIN, SERVOMAX);
  
  // Set PWM for the servo
  pwm.setPWM(servoChannels[servoIndex], 0, pulseWidth);
}

// Test function - uncomment to test individual servos
/*
void testServos() {
  Serial.println("Testing servos...");
  
  for (int servo = 0; servo < NUM_SERVOS; servo++) {
    Serial.print("Testing servo ");
    Serial.println(servo);
    
    // Move to 0 degrees
    setServoAngle(servo, 0);
    delay(1000);
    
    // Move to 90 degrees
    setServoAngle(servo, 90);
    delay(1000);
    
    // Move to 180 degrees
    setServoAngle(servo, 180);
    delay(1000);
    
    // Return to middle
    setServoAngle(servo, 90);
    delay(500);
  }
  
  Serial.println("Test complete");
}
*/
