# ESP32-CAM Dynamic Dataset Collector 

A configurable Python script for collecting facial images from an ESP32-CAM stream with customizable class configuration.

## Features
- Dynamic class configuration (up to 9 classes)
- Real-time face detection
- Configurable capture settings
- Organized dataset structure
- Simple numeric key controls (1-9)

## Setup
1. Update `ESP_IP` with your ESP32-CAM IP address
2. Configure classes in the `CLASSES` dictionary
3. Adjust parameters as needed (image format, resolution, etc.)
4. Install dependencies: `pip install opencv-python aiohttp numpy`

## Usage
Run the script and use number keys (1-9) to select classes. Each press captures up to 5 images for the selected class.

## Output
Images are saved in organized subdirectories with timestamps for uniqueness.
