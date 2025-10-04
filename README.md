# Kinda AR WhiteBoard

**Kinda AR WhiteBoard** is an interactive virtual drawing board controlled by hand gestures using OpenCV and MediaPipe. It transforms your webcam and screen into an augmented reality whiteboard that deciphers hand movements (drawing, erasing, color picking, and brush sizing) for a seamless creative experience.

## Features

- Hand gesture-based drawing: Use index finger to draw, palm to erase, pinch to resize brush.
- Color palette selection: Hover index finger on palette to change brush color.
- Smooth drawing: Kalman filter for stable lines.
- Multimode controls: Switch gestures for drawing, moving, erasing, and resizing.
- Live preview panel: See your artwork in real-time with confidence scores.
- Customizable brush: Choose color and size dynamically.
- FPS Counter: Track frame rate performance.

## Prerequisites

- Python 3.8 or later
- pip

## Installation

### 1. Clone the Repository

git clone https://github.com/CoderAmbar/kinda-ar-whiteboard.git
cd kinda-ar-whiteboard

text

### 2. Install Dependencies

Recommended method: use a virtual environment.

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt

text

**requirements.txt**
opencv-python
mediapipe
numpy

text

## Usage

Run the main script:

python kinda_whiteboard.py

text

### Controls

- **Draw**: Point with index finger.
- **Move**: Show index + middle fingers.
- **Erase**: Show open palm.
- **Resize Brush**: Pinch thumb and index together.
- **Color Change**: Hover index finger over palette.
- **Clear Canvas**: Press key `c`
- **Reset Gestures**: Press key `r`
- **Fullscreen Toggle**: Press key `f`
- **Exit Program**: Press `ESC`

On start, instructions show in the console.

## Customization

- Adjust default colors, brush sizes, and `MAX_STROKES` in `kinda_whiteboard.py`.
- To use a different webcam, change `CAMERA_ID` value at the top of the file.

## Troubleshooting

- **Camera Not Detected**: Check webcam connection, or try changing `CAMERA_ID`.
- **Laggy Performance**: Reduce `WINDOW_WIDTH`/`WINDOW_HEIGHT`, close other apps, try a faster computer.
- **MediaPipe Not Installed**: Run `pip install mediapipe`.

## File Structure

kinda-ar-whiteboard/
├── kinda_whiteboard.py
├── README.md
└── requirements.txt

text

## Credits

- MediaPipe: Hand Tracking
- OpenCV: Image and video processing
- Inspired by modern AR and creative board UI concepts

---

**Enjoy drawing on your AR WhiteBoard!**
