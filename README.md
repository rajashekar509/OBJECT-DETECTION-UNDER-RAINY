# Object Detection Under Rainy Conditions for Autonomous Vehicles

This project aims to improve object detection performance in rainy conditions using deraining techniques.

## File Structure
- `data/`: Contains rainy, clear, and derained images.
- `models/`: Stores pretrained model weights.
- `utils/`: Utility scripts for preprocessing, visualization, and evaluation.
- `derain.py`: Derains rainy images.
- `detect.py`: Performs object detection on derained images.
- `evaluate.py`: Evaluates object detection performance.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Derain images: `python derain.py`
3. Detect objects: `python detect.py`
4. Evaluate performance: `python evaluate.py`