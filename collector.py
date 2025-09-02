"""
ESP32-CAM Dynamic Dataset Collector 
Description: Collects facial images with configurable classes and controlled capture count
"""

import cv2
import numpy as np
import aiohttp
import asyncio
from datetime import datetime
from pathlib import Path

# ========== CONFIGURATION ==========
ESP_IP = "192.168.X.X"  # Update with your ESP32-CAM IP

# Define your classes here (up to 9 classes)
CLASSES = {
    "1": "person_a",  # Class 1 - will be activated with key '1'
    "2": "person_b",  # Class 2 - will be activated with key '2'
    "3": "person_c",  # Class 3 - will be activated with key '3'
    # Add more classes as needed (up to 9)
}

MAX_CAPTURE_PER_SESSION = 5  # Maximum images per capture session
IMAGE_FORMAT = "jpg"  # Output image format (jpg, png, etc.)
DATASET_PATH = "dataset"  # Root directory for dataset

# Camera stream configuration
RESOLUTION = (96, 96)  # Output image resolution
BUFFER_SIZE = 4096  # Stream buffer size
MIN_FACE_SIZE = (60, 60)  # Minimum face size for detection
# ===================================

class DatasetCollector:
    def __init__(self, esp_ip, classes, dataset_path=DATASET_PATH):
        self.stream_url = f'http://{esp_ip}/stream'
        self.dataset_path = Path(dataset_path)
        self.classes = classes
        self.current_class = None
        self.frame_buffer = b''
        self.image_counter = {cls_id: 0 for cls_id in self.classes}
        self.running = True
        self.max_capture = MAX_CAPTURE_PER_SESSION
        self.capture_count = 0
        self.image_format = IMAGE_FORMAT
        
        # Create dataset directories
        self._create_directories()
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _create_directories(self):
        """Create dataset directory structure for all classes"""
        for cls_id, cls_name in self.classes.items():
            (self.dataset_path/cls_name).mkdir(parents=True, exist_ok=True)
            # Count existing images
            pattern = f"*.{self.image_format}"
            self.image_counter[cls_id] = len(list((self.dataset_path/cls_name).glob(pattern)))

    async def _stream_processor(self):
        """Async stream processor with face extraction"""
        async with aiohttp.ClientSession() as session:
            async with session.get(self.stream_url) as response:
                while self.running:
                    try:
                        chunk = await response.content.read(self.buffer_size)
                        if not chunk:
                            break
                            
                        self.frame_buffer += chunk
                        a = self.frame_buffer.find(b'\xff\xd8')
                        b = self.frame_buffer.find(b'\xff\xd9')

                        if a != -1 and b != -1 and b > a:
                            jpg = self.frame_buffer[a:b+2]
                            self.frame_buffer = self.frame_buffer[b+2:]
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                            if frame is not None:
                                self._process_frame(frame)

                    except Exception as e:
                        print(f"Stream error: {e}")
                        break

    def _process_frame(self, frame):
        """Face detection and image processing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE
        )
        
        # Draw UI overlay
        display_frame = frame.copy()
        
        # Display current class and capture status
        if self.current_class:
            cls_name = self.classes[self.current_class]
            status_text = f"Class: {cls_name} ({self.capture_count}/{self.max_capture})"
        else:
            status_text = "Press 1-9 to select a class"
            
        cv2.putText(display_frame, status_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display class mapping
        y_offset = 60
        for i, (cls_id, cls_name) in enumerate(self.classes.items()):
            cv2.putText(display_frame, f"{cls_id}: {cls_name}",
                        (10, y_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add quit instruction
        cv2.putText(display_frame, "Q: Quit",
                    (10, y_offset + len(self.classes)*25 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save face if class selected and under capture limit
        if self.current_class and len(faces) > 0 and self.capture_count < self.max_capture:
            self._save_face(frame, faces[0])
        
        cv2.imshow("ESP32-CAM Dataset Collector", display_frame)
        key = cv2.waitKey(1)
        self._handle_key_input(key)

    def _save_face(self, frame, face_coords):
        """Save processed face image to dataset"""
        x, y, w, h = face_coords
        face_img = frame[y:y+h, x:x+w]
        
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, RESOLUTION)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        cls_name = self.classes[self.current_class]
        filename = f"{cls_name}_{timestamp}.{self.image_format}"
        save_path = self.dataset_path/cls_name/filename
        
        cv2.imwrite(str(save_path), face_img)
        self.image_counter[self.current_class] += 1
        self.capture_count += 1
        print(f"Saved {save_path} | Total: {self.image_counter[self.current_class]}")

    def _handle_key_input(self, key):
        """Handle keyboard input for class selection"""
        key_char = chr(key).lower() if key != -1 else None
        
        # Check if key matches any class ID
        if key_char in self.classes:
            self.current_class = key_char
            self.capture_count = 0
            cls_name = self.classes[key_char]
            print(f"Started capturing {cls_name.upper()} ({self.max_capture} images max)")
        elif key == ord('q'):
            self.running = False
            print("Exiting...")

    async def start(self):
        """Main collection loop"""
        print("Starting dataset collection...")
        print("Available classes:")
        for cls_id, cls_name in self.classes.items():
            print(f"  {cls_id}: {cls_name}")
        print("Press corresponding number to start capture, Q to quit")
        
        try:
            await self._stream_processor()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cv2.destroyAllWindows()
            print("\nCollection completed")
            print("Final counts:")
            for cls_id, count in self.image_counter.items():
                print(f"  {self.classes[cls_id]}: {count}")

if __name__ == "__main__":
    collector = DatasetCollector(ESP_IP, CLASSES)
    
    try:
        asyncio.run(collector.start())
    except KeyboardInterrupt:
        collector.running = False