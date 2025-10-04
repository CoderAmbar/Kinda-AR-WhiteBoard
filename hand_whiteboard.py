import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import math

# Configuration
CAMERA_ID = 0
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
BACKGROUND_COLOR = (38, 42, 55)
DEFAULT_BRUSH_COLOR = (255, 255, 255)
DEFAULT_BRUSH_SIZE = 6
MIN_BRUSH_SIZE = 1
MAX_BRUSH_SIZE = 36
MAX_STROKES = 4096

# Color palette for the UI
COLOR_PALETTE = [
    ((30, 30), (90, 90), (250, 250, 255)),   # White
    ((110, 30), (170, 90), (248, 89, 102)),  # Red
    ((190, 30), (250, 90), (95, 230, 156)),  # Green
    ((270, 30), (330, 90), (66, 152, 241)),  # Blue
    ((350, 30), (410, 90), (239, 224, 86)),  # Yellow
    ((430, 30), (490, 90), (180, 180, 180)), # Gray
]

PALETTE_HOVER_TIME = 0.33
HAND_DETECTION_CONFIDENCE = 0.8
HAND_TRACKING_CONFIDENCE = 0.8

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
FINGER_TIP_IDS = [4, 8, 12, 16, 20]

class SmoothingFilter:
    """Smooths hand movements for better drawing experience"""
    def __init__(self, process_noise=1e-4, measurement_noise=5e-2):
        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        self.filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        
        cv2.setIdentity(self.filter.processNoiseCov, process_noise)
        cv2.setIdentity(self.filter.measurementNoiseCov, measurement_noise)
        cv2.setIdentity(self.filter.errorCovPost, 1.0)
        
        self.initialized = False
        self.last_position = None

    def update(self, x, y):
        if x is None or y is None:
            return self.last_position

        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        
        if not self.initialized:
            self.filter.statePost = np.array([[x], [y], [0.0], [0.0]], np.float32)
            self.initialized = True

        self.filter.correct(measurement)
        prediction = self.filter.predict()
        self.last_position = (int(prediction[0]), int(prediction[1]))
        return self.last_position

    def reset(self):
        self.initialized = False
        self.last_position = None

def distance_between_points(point1, point2):
    """Calculate distance between two points"""
    if point1 is None or point2 is None:
        return float('inf')
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def get_finger_states(hand_landmarks, hand_side='Right'):
    """Detect which fingers are raised"""
    landmarks = hand_landmarks.landmark
    finger_states = []
    
    # Thumb detection (different for left/right hands)
    if hand_side == 'Right':
        thumb_up = landmarks[4].x < landmarks[3].x
    else:
        thumb_up = landmarks[4].x > landmarks[3].x
    finger_states.append(1 if thumb_up else 0)
    
    # Other fingers (index, middle, ring, pinky)
    for finger_id in range(1, 5):
        tip_id = FINGER_TIP_IDS[finger_id]
        pip_id = tip_id - 2  # PIP joint is two landmarks before tip
        
        finger_up = landmarks[tip_id].y < landmarks[pip_id].y
        finger_states.append(1 if finger_up else 0)
    
    return finger_states

def get_palm_center(hand_landmarks, width, height):
    """Calculate the center of the palm"""
    landmarks = hand_landmarks.landmark
    palm_points = [0, 5, 9, 13, 17]  # Key palm landmarks
    
    x_coords = [int(landmarks[i].x * width) for i in palm_points]
    y_coords = [int(landmarks[i].y * height) for i in palm_points]
    
    center_x = sum(x_coords) // len(x_coords)
    center_y = sum(y_coords) // len(y_coords)
    
    return (center_x, center_y)

class GestureDetector:
    """Recognizes hand gestures for drawing and erasing"""
    def __init__(self):
        self.recent_gestures = deque(maxlen=8)
        self.current_gesture = "NO_HAND"
        self.confidence = 0.0
        self.stable_frames = 0
        
    def detect_gesture(self, hand_landmarks, hand_side, hand_quality):
        if hand_landmarks is None or hand_quality < 0.4:
            self.current_gesture = "NO_HAND"
            self.confidence = 0.0
            self.stable_frames = 0
            return self.current_gesture, self.confidence
        
        fingers = get_finger_states(hand_landmarks, hand_side)
        raised_fingers = sum(fingers)
        
        possible_gestures = []
        gesture_confidences = []
        
        # Drawing gesture (index finger only)
        if fingers[1] == 1 and sum(fingers[2:]) == 0 and fingers[0] == 0:
            possible_gestures.append("DRAW")
            gesture_confidences.append(hand_quality * 0.9)
        
        # Move gesture (index + middle fingers)
        if fingers[1] == 1 and fingers[2] == 1 and sum(fingers[3:]) == 0 and fingers[0] == 0:
            possible_gestures.append("MOVE")
            gesture_confidences.append(hand_quality * 0.85)
        
        # Erase gesture (all fingers up)
        if raised_fingers == 5:
            possible_gestures.append("ERASE")
            gesture_confidences.append(hand_quality * 0.95)
        
        # Pinch gesture (thumb and index close together)
        thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
        index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
        pinch_distance = np.linalg.norm(thumb_tip - index_tip)
        
        if pinch_distance < 0.05 and raised_fingers <= 2:
            possible_gestures.append("PINCH")
            gesture_confidences.append((1 - pinch_distance / 0.05) * hand_quality)
        
        # Default to unknown if no clear gesture
        if not possible_gestures:
            possible_gestures.append("UNKNOWN")
            gesture_confidences.append(hand_quality * 0.5)
        
        # Pick the most likely gesture
        best_match_index = np.argmax(gesture_confidences)
        detected_gesture = possible_gestures[best_match_index]
        gesture_confidence = gesture_confidences[best_match_index]
        
        # Update gesture history
        self.recent_gestures.append((detected_gesture, gesture_confidence))
        
        # Check for consistent gesture detection
        if len(self.recent_gestures) >= 3:
            high_confidence_gestures = [g for g, c in list(self.recent_gestures) if c > 0.65]
            
            if len(high_confidence_gestures) >= 3:
                if all(g == high_confidence_gestures[0] for g in high_confidence_gestures[:3]):
                    if self.current_gesture == high_confidence_gestures[0]:
                        self.stable_frames += 1
                    else:
                        self.stable_frames = 1
                    
                    self.current_gesture = high_confidence_gestures[0]
                    self.confidence = np.mean([c for g, c in self.recent_gestures if g == self.current_gesture])
                else:
                    self.stable_frames = max(0, self.stable_frames - 1)
            else:
                self.stable_frames = max(0, self.stable_frames - 1)
        else:
            self.current_gesture = detected_gesture
            self.confidence = gesture_confidence
        
        # Reduce confidence for unstable gestures
        if self.stable_frames < 2 and self.current_gesture != "NO_HAND":
            self.confidence *= 0.7
        
        return self.current_gesture, min(1.0, self.confidence)

def draw_color_palette(image, selected_color_index):
    """Draw the color selection palette"""
    for i, (top_left, bottom_right, color) in enumerate(COLOR_PALETTE):
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2
        
        cv2.circle(image, (center_x, center_y), 28, color, -1)
        
        if i == selected_color_index:
            cv2.circle(image, (center_x, center_y), 32, (255, 255, 255), 2)
    
    cv2.putText(image, "Colors", (30, 105), cv2.FONT_HERSHEY_DUPLEX, 0.8, (210, 210, 210), 2)
    cv2.line(image, (30, 112), (490, 112), (100, 100, 120), 2)

def draw_drawing_cursor(image, position, color=(0, 255, 120)):
    """Draw the brush cursor"""
    cv2.circle(image, position, 14, color, 18, cv2.LINE_AA)
    cv2.circle(image, position, 6, color, -1, cv2.LINE_AA)

def draw_eraser_cursor(image, position, size):
    """Draw the eraser cursor"""
    cv2.circle(image, position, size, (255, 100, 100), 3, cv2.LINE_AA)
    cv2.circle(image, position, size-2, (255, 150, 150), 1, cv2.LINE_AA)
    cv2.circle(image, position, 4, (255, 100, 100), -1, cv2.LINE_AA)

def draw_info_panel(image, gesture, confidence, color, brush_size):
    """Draw the information panel at the bottom"""
    # Create semi-transparent background
    overlay = image.copy()
    panel_top_left = (20, WINDOW_HEIGHT - 130)
    panel_bottom_right = (600, WINDOW_HEIGHT - 30)
    
    cv2.rectangle(overlay, panel_top_left, panel_bottom_right, (26, 26, 40), -1)
    
    alpha = 0.41
    image_section = image[panel_top_left[1]:panel_bottom_right[1], panel_top_left[0]:panel_bottom_right[0]]
    overlay_section = overlay[panel_top_left[1]:panel_bottom_right[1], panel_top_left[0]:panel_bottom_right[0]]
    
    image[panel_top_left[1]:panel_bottom_right[1], panel_top_left[0]:panel_bottom_right[0]] = \
        cv2.addWeighted(image_section, 1-alpha, overlay_section, alpha, 0)
    
    # Confidence color (green=high, blue=medium, red=low)
    if confidence > 0.7:
        confidence_color = (100, 255, 100)
    elif confidence > 0.5:
        confidence_color = (100, 200, 255)
    else:
        confidence_color = (255, 100, 100)
    
    # Display information
    cv2.putText(image, f'Gesture: {gesture}', (40, WINDOW_HEIGHT - 105), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (240, 255, 240), 2)
    
    cv2.putText(image, f'Confidence: {confidence:.2f}', (40, WINDOW_HEIGHT - 80), 
                cv2.FONT_HERSHEY_PLAIN, 1.2, confidence_color, 2)
    
    cv2.putText(image, f'Brush: {color} | {brush_size}px', (40, WINDOW_HEIGHT - 60), 
                cv2.FONT_HERSHEY_PLAIN, 1.2, (180, 220, 230), 2)
    
    cv2.putText(image, 'Index: Draw | Index+Middle: Move | Palm: Erase | Pinch: Size', 
                (40, WINDOW_HEIGHT - 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (140, 160, 180), 1)

def draw_hand_skeleton(image, hand_landmarks, confidence):
    """Draw the hand skeleton with confidence-based visibility"""
    if confidence > 0.5:
        landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
        connection_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                 landmark_style, connection_style)

class DrawingManager:
    """Manages all drawing and erasing operations"""
    def __init__(self, max_strokes=MAX_STROKES):
        self.all_strokes = deque(maxlen=max_strokes)
        self.current_stroke = []
        self.current_erase = []
        
    def add_point(self, point, is_erasing=False):
        if is_erasing:
            if point is not None:
                self.current_erase.append(point)
            else:
                if self.current_erase:
                    self.apply_erasing()
                    self.current_erase.clear()
        else:
            if point is not None:
                self.current_stroke.append(point)
            else:
                if self.current_stroke:
                    self.all_strokes.append(self.current_stroke.copy())
                    self.current_stroke.clear()
    
    def apply_erasing(self):
        """Permanently remove strokes in the erased area"""
        if len(self.current_erase) < 2:
            return
            
        # Create erase area mask
        erase_mask = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
        erase_size = 40
        
        # Draw erase path on mask
        for i in range(1, len(self.current_erase)):
            cv2.line(erase_mask, self.current_erase[i-1], self.current_erase[i], 
                    255, erase_size, cv2.LINE_AA)
        
        # Filter out strokes that intersect with erase area
        updated_strokes = deque(maxlen=MAX_STROKES)
        
        for stroke in self.all_strokes:
            if len(stroke) < 2:
                updated_strokes.append(stroke)
                continue
                
            # Check each stroke segment
            clean_stroke = []
            for i in range(1, len(stroke)):
                point1, point2 = stroke[i-1], stroke[i]
                
                # Test multiple points along the segment
                segment_erased = False
                segment_length = max(1, int(distance_between_points(point1, point2)))
                
                for t in np.linspace(0, 1, max(5, segment_length // 10)):
                    x = int(point1[0] * (1-t) + point2[0] * t)
                    y = int(point1[1] * (1-t) + point2[1] * t)
                    
                    if 0 <= x < WINDOW_WIDTH and 0 <= y < WINDOW_HEIGHT:
                        if erase_mask[y, x] > 0:
                            segment_erased = True
                            break
                
                if not segment_erased:
                    if not clean_stroke:
                        clean_stroke.append(point1)
                    clean_stroke.append(point2)
                else:
                    # Save the unerased part and start new segment
                    if len(clean_stroke) >= 2:
                        updated_strokes.append(clean_stroke.copy())
                    clean_stroke = []
            
            if len(clean_stroke) >= 2:
                updated_strokes.append(clean_stroke)
        
        self.all_strokes = updated_strokes
    
    def render_drawing(self, canvas, color, brush_size, background_color):
        """Render all strokes to the canvas"""
        # Clear canvas
        canvas[:] = background_color
        
        # Draw all saved strokes
        for stroke in self.all_strokes:
            for i in range(1, len(stroke)):
                cv2.line(canvas, stroke[i-1], stroke[i], color, brush_size, cv2.LINE_AA)
        
        # Draw current stroke in progress
        for i in range(1, len(self.current_stroke)):
            cv2.line(canvas, self.current_stroke[i-1], self.current_stroke[i], 
                    color, brush_size, cv2.LINE_AA)
        
        # Show current erase area (visual feedback only)
        erase_size = 40
        for i in range(1, len(self.current_erase)):
            cv2.line(canvas, self.current_erase[i-1], self.current_erase[i], 
                    background_color, erase_size, cv2.LINE_AA)
    
    def clear_canvas(self):
        """Clear all drawing data"""
        self.all_strokes.clear()
        self.current_stroke.clear()
        self.current_erase.clear()

def main():
    """Main application loop"""
    # Initialize camera
    camera = cv2.VideoCapture(CAMERA_ID)
    if not camera.isOpened():
        # Try alternative camera IDs
        for cam_id in range(1, 5):
            camera = cv2.VideoCapture(cam_id)
            if camera.isOpened():
                print(f"Using camera {cam_id}")
                break
    
    if not camera.isOpened():
        print("Could not access camera")
        return
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    
    # Initialize drawing surface
    drawing_surface = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    drawing_surface[:] = BACKGROUND_COLOR
    
    # Initialize managers
    drawing_manager = DrawingManager()
    finger_smoother = SmoothingFilter()
    palm_smoother = SmoothingFilter()
    gesture_detector = GestureDetector()
    
    # Application state
    current_color = DEFAULT_BRUSH_COLOR
    selected_color_index = 0
    brush_size = DEFAULT_BRUSH_SIZE
    palette_hover_start = None
    last_hovered_color = None
    last_hand_detected = time.time()
    
    print("Hand Drawing Board Started")
    print("Instructions:")
    print("  - Point with index finger to draw")
    print("  - Show index and middle fingers to move")
    print("  - Open palm to erase")
    print("  - Pinch thumb and index to adjust brush size")
    print("  - Hover over color palette to change color")
    print("  - Press 'c' to clear canvas")
    print("  - Press 'r' to reset gesture detection")
    print("  - Press ESC to exit")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
        model_complexity=1
    ) as hand_detector:
        
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read camera frame")
                break
            
            # Mirror the frame for more intuitive interaction
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            # Process frame for hand detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_results = hand_detector.process(rgb_frame)
            
            # Reset state for this frame
            current_gesture = 'No hand detected'
            gesture_confidence = 0.0
            index_position = None
            thumb_position = None
            palm_position = None
            
            if detection_results.multi_hand_landmarks and detection_results.multi_handedness:
                last_hand_detected = time.time()
                
                try:
                    hand_landmarks = detection_results.multi_hand_landmarks[0]
                    hand_side = detection_results.multi_handedness[0].classification[0].label
                    
                    # Detect current gesture
                    current_gesture, gesture_confidence = gesture_detector.detect_gesture(
                        hand_landmarks, hand_side, 0.8
                    )

                    # Get finger positions
                    index_tip = hand_landmarks.landmark[8]
                    index_x, index_y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
                    index_position = (index_x, index_y)
                    
                    thumb_tip = hand_landmarks.landmark[4]
                    thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
                    thumb_position = (thumb_x, thumb_y)
                    
                    # Get palm position
                    palm_x, palm_y = get_palm_center(hand_landmarks, frame_width, frame_height)
                    palm_position = (palm_x, palm_y)
                    
                    # Apply smoothing for stable drawing
                    if gesture_confidence > 0.6:
                        smooth_index = finger_smoother.update(index_x, index_y)
                        smooth_palm = palm_smoother.update(palm_x, palm_y)
                    else:
                        smooth_index = (index_x, index_y)
                        smooth_palm = (palm_x, palm_y)
                    
                    # Calculate eraser size based on screen dimensions
                    screen_diagonal = math.hypot(frame_width, frame_height)
                    eraser_size = int(max(brush_size * 2, screen_diagonal * 0.049))

                    # Handle different gestures
                    if current_gesture == 'DRAW' and gesture_confidence > 0.65:
                        drawing_manager.add_point(smooth_index, is_erasing=False)
                    elif current_gesture == 'MOVE' and gesture_confidence > 0.65:
                        drawing_manager.add_point(None, is_erasing=False)
                    elif current_gesture == 'ERASE' and gesture_confidence > 0.65:
                        drawing_manager.add_point(smooth_palm, is_erasing=True)
                    else:
                        drawing_manager.add_point(None, is_erasing=False)
                        drawing_manager.add_point(None, is_erasing=True)

                    # Draw hand visualization
                    draw_hand_skeleton(frame, hand_landmarks, gesture_confidence)
                    
                    # Show appropriate cursor
                    if current_gesture == 'DRAW' and smooth_index and gesture_confidence > 0.65:
                        draw_drawing_cursor(frame, smooth_index)
                    elif current_gesture == 'ERASE' and smooth_palm and gesture_confidence > 0.65:
                        draw_eraser_cursor(frame, smooth_palm, eraser_size)

                    # Handle color palette interaction
                    if gesture_confidence > 0.65 and index_position:
                        for i, (top_left, bottom_right, color) in enumerate(COLOR_PALETTE):
                            if (top_left[0] <= index_position[0] <= bottom_right[0] and 
                                top_left[1] <= index_position[1] <= bottom_right[1]):
                                
                                if last_hovered_color != i:
                                    palette_hover_start = time.time()
                                    last_hovered_color = i
                                else:
                                    if time.time() - (palette_hover_start or 0) > PALETTE_HOVER_TIME:
                                        current_color = color
                                        selected_color_index = i
                                        palette_hover_start = None
                                        last_hovered_color = None
                                break
                        else:
                            last_hovered_color = None
                            palette_hover_start = None

                    # Adjust brush size with pinch gesture
                    if thumb_position and index_position and gesture_confidence > 0.6:
                        pinch_distance = distance_between_points(thumb_position, index_position)
                        screen_diagonal = math.hypot(frame_width, frame_height)
                        
                        new_size = np.interp(pinch_distance, [8, screen_diagonal * 0.22], 
                                           [MIN_BRUSH_SIZE, MAX_BRUSH_SIZE])
                        brush_size = int(max(MIN_BRUSH_SIZE, min(MAX_BRUSH_SIZE, new_size)))
                        
                except Exception as error:
                    print(f"Error processing hand: {error}")
                    current_gesture = "Processing error"
                    gesture_confidence = 0.0
            else:
                # No hand detected - end current strokes
                drawing_manager.add_point(None, is_erasing=False)
                drawing_manager.add_point(None, is_erasing=True)
                
                # Reset if no hand for a while
                if time.time() - last_hand_detected > 2.0:
                    gesture_detector = GestureDetector()
                    finger_smoother.reset()
                    palm_smoother.reset()

            # Update drawing surface
            drawing_manager.render_drawing(drawing_surface, current_color, brush_size, BACKGROUND_COLOR)

            # Combine camera feed with drawing
            combined_frame = frame.copy()
            transparency = 0.91
            drawing_areas = (drawing_surface != np.array(BACKGROUND_COLOR)).any(axis=2)
            combined_frame[drawing_areas] = (
                combined_frame[drawing_areas] * (1-transparency) + 
                drawing_surface[drawing_areas] * transparency
            ).astype(np.uint8)

            # Draw UI elements
            draw_color_palette(combined_frame, selected_color_index)
            draw_info_panel(combined_frame, current_gesture, gesture_confidence, current_color, brush_size)
            
            # Show drawing preview
            preview = cv2.resize(drawing_surface, (320, 180))
            combined_frame[15:195, WINDOW_WIDTH-335:WINDOW_WIDTH-15] = preview

            # Show frame rate
            fps = camera.get(cv2.CAP_PROP_FPS)
            cv2.putText(combined_frame, f'FPS: {fps:.1f}', (WINDOW_WIDTH-150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Hand Drawing Board', combined_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('c'):
                drawing_surface[:] = BACKGROUND_COLOR
                drawing_manager.clear_canvas()
                print("Canvas cleared")
            elif key == ord('r'):
                gesture_detector = GestureDetector()
                finger_smoother.reset()
                palm_smoother.reset()
                print("Gesture detection reset")
            elif key == ord('f'):
                # Toggle fullscreen
                current_state = cv2.getWindowProperty('Hand Drawing Board', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('Hand Drawing Board', cv2.WND_PROP_FULLSCREEN, not current_state)

    # Clean up
    camera.release()
    cv2.destroyAllWindows()
    print("Application closed")

if __name__ == '__main__':
    main()