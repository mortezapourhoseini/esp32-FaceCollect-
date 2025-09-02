"""
ESP32-CAM Dataset Collector 
Description: Collects facial images with controlled capture count
"""

import cv2
import numpy as np
import aiohttp
import asyncio
from datetime import datetime
from pathlib import Path

ESP_IP = "192.168.X.X"  # Update with your ESP32-CAM IP

class DatasetCollector:
    def __init__(self, esp_ip, dataset_path="dataset"):
        self.stream_url = f'http://{esp_ip}/stream'
        self.dataset_path = Path(dataset_path)
        self.classes = ["morteza", "others"]
        self.current_class = None
        self.frame_buffer = b''
        self.image_counter = {cls: 0 for cls in self.classes}
        self.running = True
        self.max_capture = 5  # Maximum images per capture session
        self.capture_count = 0
        
        # Create dataset directories
        self._create_directories()
        
        # Camera stream configuration
        self.resolution = (96, 96)
        self.buffer_size = 4096
        self.min_face_size = (60, 60)
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _create_directories(self):
        """Create dataset directory structure"""
        for cls in self.classes:
            (self.dataset_path/cls).mkdir(parents=True, exist_ok=True)
            self.image_counter[cls] = len(list((self.dataset_path/cls).glob("*.jpg")))

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
            minSize=self.min_face_size
        )
        
        # Draw UI overlay
        display_frame = frame.copy()
        status_text = f"Class: {self.current_class or 'None'} ({self.capture_count}/{self.max_capture})"
        cv2.putText(display_frame, status_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "M: Morteza | O: Others | Q: Quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
        face_img = cv2.resize(face_img, self.resolution)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"{self.current_class}_{timestamp}.jpg"
        save_path = self.dataset_path/self.current_class/filename
        
        cv2.imwrite(str(save_path), face_img)
        self.image_counter[self.current_class] += 1
        self.capture_count += 1
        print(f"Saved {save_path} | Total: {self.image_counter[self.current_class]}")

    def _handle_key_input(self, key):
        """Handle keyboard input"""
        if key == ord('m'):
            self.current_class = "morteza"
            self.capture_count = 0
            print("Started capturing MORTEZA (5 images max)")
        elif key == ord('o'):
            self.current_class = "others"
            self.capture_count = 0
            print("Started capturing OTHERS (5 images max)")
        elif key == ord('q'):
            self.running = False
            print("Exiting...")

    async def start(self):
        """Main collection loop"""
        print("Starting dataset collection...")
        print("Press M/O to start capture (5 images each), Q to quit")
        
        try:
            await self._stream_processor()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cv2.destroyAllWindows()
            print("\nCollection completed")
            print(f"Final counts: {self.image_counter}")

if __name__ == "__main__":
    
    collector = DatasetCollector(ESP_IP)
    
    try:
        asyncio.run(collector.start())
    except KeyboardInterrupt:
        collector.running = False
