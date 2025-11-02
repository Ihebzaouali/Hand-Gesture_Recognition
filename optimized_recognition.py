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
"""Optimized gesture recognition for Raspberry Pi."""

import argparse
import sys
import time
import threading
from collections import deque

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
frame_buffer = deque(maxlen=2)  # Small buffer to avoid processing delays
processing_frame = False

def run(model: str, num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
    """Optimized gesture recognition for Raspberry Pi."""
    
    global processing_frame
    
    # Reduced resolution for better performance
    optimized_width = min(width, 320)
    optimized_height = min(height, 240)
    
    # Start capturing video input
    cap = cv2.VideoCapture('http://192.168.1.12:4747/video_feed')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, optimized_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, optimized_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    # Optimized visualization parameters
    row_size = 30
    left_margin = 10
    text_color = (0, 255, 0)  # Green for better visibility
    font_size = 0.6
    font_thickness = 1
    fps_avg_frame_count = 10
    
    # Simplified label parameters
    label_text_color = (255, 255, 255)
    label_font_size = 0.7
    label_thickness = 2
    
    recognition_result_list = []
    last_gesture = ""
    gesture_confidence = 0
    frame_skip_counter = 0
    
    def save_result(result: vision.GestureRecognizerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, processing_frame
        
        # Calculate FPS less frequently
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()
        
        recognition_result_list.append(result)
        COUNTER += 1
        processing_frame = False
    
    # Initialize gesture recognizer with optimized settings
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=num_hands,
        min_hand_detection_confidence=max(min_hand_detection_confidence, 0.7),  # Higher threshold
        min_hand_presence_confidence=max(min_hand_presence_confidence, 0.6),
        min_tracking_confidence=max(min_tracking_confidence, 0.6),
        result_callback=save_result
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
    
    print(f"Starting optimized gesture recognition at {optimized_width}x{optimized_height}")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read frame, retrying...")
            time.sleep(0.1)
            continue
        
        # Skip frames to reduce processing load
        frame_skip_counter += 1
        if frame_skip_counter % 2 != 0:  # Process every 2nd frame
            continue
            
        # Only process if not currently processing
        if not processing_frame:
            processing_frame = True
            
            # Flip image horizontally for mirror effect
            image = cv2.flip(image, 1)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Run gesture recognition asynchronously
            try:
                recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)
            except:
                processing_frame = False
        
        # Display FPS
        fps_text = f'FPS: {FPS:.1f}'
        cv2.putText(image, fps_text, (left_margin, row_size), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
        
        # Process results if available
        if recognition_result_list:
            result = recognition_result_list[0]
            
            # Draw results for each detected hand
            if result.hand_landmarks and result.gestures:
                for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
                    if hand_index < len(result.gestures):
                        # Get gesture info
                        gesture = result.gestures[hand_index][0]
                        gesture_name = gesture.category_name
                        confidence = round(gesture.score, 2)
                        
                        # Update gesture info (with smoothing)
                        if confidence > 0.7:  # Only show high-confidence gestures
                            last_gesture = gesture_name
                            gesture_confidence = confidence
                        
                        # Calculate hand bounding box (simplified)
                        h, w = image.shape[:2]
                        x_min = int(min([lm.x for lm in hand_landmarks]) * w)
                        y_min = int(min([lm.y for lm in hand_landmarks]) * h)
                        
                        # Display gesture text
                        gesture_text = f'{last_gesture} ({gesture_confidence})'
                        text_y = max(y_min - 10, 20)
                        cv2.putText(image, gesture_text, (x_min, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, label_font_size,
                                   label_text_color, label_thickness)
                        
                        # Draw simplified hand landmarks (every 4th landmark for performance)
                        for i, landmark in enumerate(hand_landmarks):
                            if i % 4 == 0:  # Draw every 4th landmark
                                x = int(landmark.x * w)
                                y = int(landmark.y * h)
                                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            
            recognition_result_list.clear()
        
        # Show the frame
        cv2.imshow('Optimized Gesture Recognition', image)
        
        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Cleanup
    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()

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
    parser.add_argument('--cameraId', help='Camera ID', 
                       required=False, default=0, type=int)
    parser.add_argument('--frameWidth', help='Frame width', 
                       required=False, default=320, type=int)  # Reduced default
    parser.add_argument('--frameHeight', help='Frame height', 
                       required=False, default=240, type=int)  # Reduced default
    
    args = parser.parse_args()
    
    print("=== Optimized MediaPipe Gesture Recognition ===")
    print(f"Resolution: {args.frameWidth}x{args.frameHeight}")
    print(f"Max hands: {args.numHands}")
    print("Press ESC to exit")
    
    run(args.model, args.numHands, args.minHandDetectionConfidence,
        args.minHandPresenceConfidence, args.minTrackingConfidence,
        args.cameraId, args.frameWidth, args.frameHeight)

if __name__ == '__main__':
    main()