# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optimized gesture recognition for Raspberry Pi with servo control."""

import argparse
import sys
import time
import threading
import json
from collections import deque
import serial
import math

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables for optimization
COUNTER, FPS = 0, 0
START_TIME = time.time()
frame_buffer = deque(maxlen=2)
processing_frame = False

# Servo control variables
arduino_serial = None
last_servo_command = None
servo_command_cooldown = 0.1  # Minimum time between servo commands

# Finger landmark indices for MediaPipe
FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
FINGER_PIPS = [3, 6, 10, 14, 18]  # Previous joints for angle calculation

def init_arduino_connection(port='/dev/ttyACM0', baudrate=9600): # Replace /dev/ttyACM0 with your actual Arduino port
    """Initialize serial connection to Arduino."""
    global arduino_serial
    try:
        arduino_serial = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize
        print(f"Arduino connected on {port}")
        return True
    except Exception as e:
        print(f"Failed to connect to Arduino: {e}")
        return False

def calculate_finger_angles(hand_landmarks):
    """Calculate finger angles from hand landmarks."""
    angles = []
    
    for i in range(5):  # 5 fingers
        if i == 0:  # Thumb - special case
            # Calculate thumb angle based on tip and MCP joint
            tip = hand_landmarks[FINGER_TIPS[i]]
            mcp = hand_landmarks[2]  # Thumb MCP
            
            # Simple angle calculation for thumb
            angle = math.atan2(tip.y - mcp.y, tip.x - mcp.x)
            angle = math.degrees(angle)
            # Normalize to 0-180 range for servo
            angle = max(0, min(180, (angle + 90) % 180))
            
        else:  # Other fingers
            tip = hand_landmarks[FINGER_TIPS[i]]
            pip = hand_landmarks[FINGER_PIPS[i]]
            mcp = hand_landmarks[FINGER_PIPS[i] - 1]
            
            # Calculate angle between segments
            v1 = [pip.x - mcp.x, pip.y - mcp.y]
            v2 = [tip.x - pip.x, tip.y - pip.y]
            
            # Calculate angle
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = math.degrees(math.acos(cos_angle))
                
                # Convert to servo range (0-180)
                # Straight finger = 180, bent finger = 0
                angle = 180 - angle
            else:
                angle = 90  # Default position
            
            angle = max(0, min(180, angle))
        
        angles.append(int(angle))
    
    return angles

def send_servo_command(angles):
    """Send servo angles to Arduino."""
    global arduino_serial, last_servo_command
    
    if arduino_serial is None:
        return False
    
    # Check cooldown
    current_time = time.time()
    if (last_servo_command is not None and 
        current_time - last_servo_command < servo_command_cooldown):
        return False
    
    try:
        # Format: "S,angle1,angle2,angle3,angle4,angle5\n"
        command = f"S,{','.join(map(str, angles))}\n"
        arduino_serial.write(command.encode())
        last_servo_command = current_time
        return True
    except Exception as e:
        print(f"Error sending servo command: {e}")
        return False

def gesture_to_servo_positions(gesture_name):
    """Convert gesture name to predefined servo positions."""
    gesture_positions = {
        'Open_Palm': [90, 90, 90, 90, 90],      # Open hand
        'Closed_Fist': [180, 180, 180, 180, 180],       # Closed fist
        'Pointing_Up': [180, 180, 90, 180, 180],      # Index finger up
        'Thumb_Up': [90, 180, 180, 180, 180],         # Thumb up
        'Victory': [180, 180, 90, 90, 180],         # Peace sign
        'ILoveYou': [90, 90, 90, 180, 180],       # I love you sign
        'None': [20, 20, 20, 20, 20]               # Neutral position
    }
    
    return gesture_positions.get(gesture_name, [90, 90, 90, 90, 90])

def run(model: str, num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int, arduino_port: str) -> None:
    """Optimized gesture recognition for Raspberry Pi with servo control."""
    
    global processing_frame
    
    # Initialize Arduino connection
    arduino_connected = init_arduino_connection(arduino_port)
    if not arduino_connected:
        print("Warning: Arduino not connected. Servo control disabled.")
    
    # Reduced resolution for better performance
    optimized_width = min(width, 320)
    optimized_height = min(height, 240)
    
    # Start capturing video input
    cap = cv2.VideoCapture('http://192.168.1.12:4747/video_feed')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, optimized_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, optimized_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Visualization parameters
    row_size = 30
    left_margin = 10
    text_color = (0, 255, 0)
    font_size = 0.6
    font_thickness = 1
    fps_avg_frame_count = 10
    
    label_text_color = (255, 255, 255)
    label_font_size = 0.7
    label_thickness = 2
    
    recognition_result_list = []
    last_gesture = "None"
    gesture_confidence = 0
    frame_skip_counter = 0
    gesture_stability_count = 0
    required_stability = 5  # Frames of stable gesture before sending command
    
    def save_result(result: vision.GestureRecognizerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, processing_frame
        
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()
        
        recognition_result_list.append(result)
        COUNTER += 1
        processing_frame = False
    
    # Initialize gesture recognizer
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=num_hands,
        min_hand_detection_confidence=max(min_hand_detection_confidence, 0.7),
        min_hand_presence_confidence=max(min_hand_presence_confidence, 0.6),
        min_tracking_confidence=max(min_tracking_confidence, 0.6),
        result_callback=save_result
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
    
    print(f"Starting gesture recognition with servo control at {optimized_width}x{optimized_height}")
    print("Arduino status:", "Connected" if arduino_connected else "Disconnected")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read frame, retrying...")
            time.sleep(0.1)
            continue
        
        frame_skip_counter += 1
        if frame_skip_counter % 2 != 0:
            continue
            
        if not processing_frame:
            processing_frame = True
            
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            try:
                recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)
            except:
                processing_frame = False
        
        # Display FPS and Arduino status
        fps_text = f'FPS: {FPS:.1f}'
        cv2.putText(image, fps_text, (left_margin, row_size), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
        
        arduino_status = "Arduino: Connected" if arduino_connected else "Arduino: Disconnected"
        cv2.putText(image, arduino_status, (left_margin, row_size * 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
        
        # Process results
        if recognition_result_list:
            result = recognition_result_list[0]
            current_gesture = "None"
            current_confidence = 0
            
            if result.hand_landmarks and result.gestures:
                for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
                    if hand_index < len(result.gestures):
                        gesture = result.gestures[hand_index][0]
                        gesture_name = gesture.category_name
                        confidence = round(gesture.score, 2)
                        
                        if confidence > 0.7:
                            current_gesture = gesture_name
                            current_confidence = confidence
                        
                        # Calculate hand bounding box
                        h, w = image.shape[:2]
                        x_min = int(min([lm.x for lm in hand_landmarks]) * w)
                        y_min = int(min([lm.y for lm in hand_landmarks]) * h)
                        
                        # Display gesture info
                        gesture_text = f'{current_gesture} ({current_confidence})'
                        text_y = max(y_min - 10, 20)
                        cv2.putText(image, gesture_text, (x_min, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, label_font_size,
                                   label_text_color, label_thickness)
                        
                        # Draw landmarks
                        for i, landmark in enumerate(hand_landmarks):
                            if i % 4 == 0:
                                x = int(landmark.x * w)
                                y = int(landmark.y * h)
                                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
                        
                        # Servo control logic
                        if arduino_connected and current_confidence > 0.7:
                            if current_gesture == last_gesture:
                                gesture_stability_count += 1
                            else:
                                gesture_stability_count = 0
                                last_gesture = current_gesture
                            
                            # Send servo command when gesture is stable
                            if gesture_stability_count >= required_stability:
                                # Option 1: Use predefined positions
                                servo_angles = gesture_to_servo_positions(current_gesture)
                                
                                # Option 2: Use calculated finger angles (uncomment to use)
                                # servo_angles = calculate_finger_angles(hand_landmarks)
                                
                                if send_servo_command(servo_angles):
                                    print(f"Sent servo command for {current_gesture}: {servo_angles}")
                                
                                gesture_stability_count = 0  # Reset after sending command
            
            recognition_result_list.clear()
        
        # Display current gesture status
        status_text = f'Gesture: {last_gesture}'
        cv2.putText(image, status_text, (left_margin, row_size * 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
        
        cv2.imshow('Gesture Recognition with Servo Control', image)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Cleanup
    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()
    if arduino_serial:
        arduino_serial.close()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Gesture recognition model path', 
                       required=False, default='gesture_recognizer.task')
    parser.add_argument('--numHands', help='Max number of hands', 
                       required=False, default=1, type=int)
    parser.add_argument('--minHandDetectionConfidence', 
                       help='Min hand detection confidence', 
                       required=False, default=0.5, type=float)
    parser.add_argument('--minHandPresenceConfidence', 
                       help='Min hand presence confidence', 
                       required=False, default=0.5, type=float)
    parser.add_argument('--minTrackingConfidence', 
                       help='Min tracking confidence', 
                       required=False, default=0.5, type=float)
    parser.add_argument('--cameraId', help='Camera ID (not used with video feed)', 
                       required=False, default=0, type=int)
    parser.add_argument('--frameWidth', help='Frame width', 
                       required=False, default=320, type=int)
    parser.add_argument('--frameHeight', help='Frame height', 
                       required=False, default=240, type=int)
    parser.add_argument('--arduinoPort', help='Arduino serial port', 
                       required=False, default='/dev/ttyACM0', type=str)
    
    args = parser.parse_args()
    
    print("=== Gesture Recognition with Servo Control ===")
    print(f"Resolution: {args.frameWidth}x{args.frameHeight}")
    print(f"Max hands: {args.numHands}")
    print(f"Arduino port: {args.arduinoPort}")
    print("Press ESC to exit")
    
    run(args.model, args.numHands, args.minHandDetectionConfidence,
        args.minHandPresenceConfidence, args.minTrackingConfidence,
        args.cameraId, args.frameWidth, args.frameHeight, args.arduinoPort)

if __name__ == '__main__':
    main()