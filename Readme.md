# ✋ Hand Gesture Recognition & Servo Control

This project implements a **hand gesture recognition system** using **MediaPipe** and an **Arduino-controlled servo system** via the **PCA9685** module.  
The system has been tested on a **Raspberry Pi 4 (4GB RAM)** using a **phone camera with DroidCam** as the video input.

---

## 🧠 Project Overview

The system performs:

- Real-time hand gesture recognition using **MediaPipe**.  
- Gesture classification with the trained model `gesture_recognition.task`.  
- Servo control via **Arduino** to reproduce hand gestures on physical servos.  
- **Gesture stability logic** to avoid false servo movements.  

> The video feed from the phone is connected over **Wi-Fi**, no additional camera setup is needed.  

---

## 🗂️ Project Structure

iheb_hand_gesture_recognition/

├── arduino/  

 └── servo_control

│    └── servo_control.ino # Arduino code to control servos via PCA9685 according to the gesture
│ └── libraires # libraires for the PCA9685 module
│
├── mediapipe_env/	
│ └── (virtual environment files)
│ └── bin/activate # Must be sourced to activate Python env
│
├── gesture_recognition.task # Trained gesture recognition model
├── recognition.py # Base gesture recognition (non-optimized)
├── optimized.py # Optimized recognition (higher FPS, cleaner)
├── optimized_control.py # Full system: gesture recognition + servo control
├── requirements.txt   # Liste des dépendances Python
└── README.md # Project documentation


---

## ⚙️ Requirements

### 🔹 Hardware
- **Raspberry Pi 4 (4/8GB RAM)**  
- Phone camera with **DroidCam** OR USB webcam (camera setup configuration is needed )
- Arduino board (Uno, Mega, etc.)  
- PCA9685 16-channel servo driver  
- Servomotors (SPT5632 or similar)  
- USB cable for Arduino communication  

### 🔹 Software
- Raspberry Pi OS / Ubuntu  
- Python 3.11+  
- Arduino IDE  
- DroidCam client on phone (no need to install on Raspberry Pi)

---

## 🧩 Python Dependencies

Le projet nécessite Python 3.11+ et les packages suivants (voir requirements.txt) :

mediapipe

opencv-python

numpy

pyserial

Installation rapide :

```bash
python3 -m venv mediapipe_env
source ~/mediapipe_env/bin/activate
pip install -r requirements.txt

Then install dependencies:

```bash
pip install mediapipe opencv-python numpy pyserial


##How to Run : 
1️⃣ Start DroidCam on your phone

-Open DroidCam on your phone and leave it running.
-Make sure the phone and Raspberry Pi are connected to the same Wi-Fi network.
-Note the video feed URL displayed by DroidCam, e.g.:
http://192.168.1.12:4747/video_feed(The script optimized_control.py already contains this URL. Adjust it based on your Wifi IP adress)

2️⃣ Activate the Python environment

```bash
cd "path_to_your_project_folder/"
source ~/mediapipe_env/bin/activate

3️⃣ Check your Arduino port

Verify with:
```bash
ls /dev/ttyACM* /dev/ttyUSB* 
Make sure to use the correct port when running the script.

4️⃣ Run the Python script

```bash
python3 optimized_control.py

-Replace /dev/ttyACM0 with your actual Arduino port.
-The script automatically connects to the phone camera via Wi-Fi.
-It detects hand gestures and sends servo commands to the Arduino.
-Press ESC to exit.

##Arduino Setup:
Upload the code in arduino/servo_control.ino via Arduino IDE.
The Arduino listens to serial commands from Python and drives the 5 servos via the PCA9685 module.

##Workflow :

-Phone camera streams video via DroidCam over Wi-Fi.

-MediaPipe extracts hand landmarks.

-Model (gesture_recognition.task) classifies the gesture.

-Python sends commands via serial to Arduino.

-Arduino moves servos according to the gesture.

-Gesture stability logic ensures only stable gestures trigger servo movement.


##Future Improvements :

-Support multiple hands simultaneously	

-Add more gestures and retrain the model

-Integrate with ROS 2 for robotic applications

-Develop GUI for visualization 

-Test on more powerful development boards (e.g., NVIDIA Jetson) to increase performance and allow real-time multi-hand recognition




















