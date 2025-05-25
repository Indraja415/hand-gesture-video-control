# Import necessary libraries
import cv2          # OpenCV for computer vision and camera operations
import mediapipe as mp    # Google's MediaPipe for hand detection and tracking
import pyautogui    # Library for automating GUI interactions (keyboard/mouse)
import time         # For time-related operations and delays

def count_fingers(lst):
    """
    Function to count the number of fingers raised
    lst: MediaPipe hand landmarks object containing 21 hand keypoints
    """
    cnt = 0  # Initialize finger counter
    
    # Calculate threshold based on hand size (distance between wrist and middle finger base)
    # This helps adapt to different hand sizes and distances from camera
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2
    
    # Check each finger by comparing fingertip position with finger base
    # If fingertip is significantly higher than base, finger is considered raised
    
    # Index finger (landmarks 5-8)
    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
        cnt += 1
    
    # Middle finger (landmarks 9-12)
    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
        cnt += 1
    
    # Ring finger (landmarks 13-16)
    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
        cnt += 1
    
    # Pinky finger (landmarks 17-20)
    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
        cnt += 1
    
    # Thumb detection (different logic due to thumb orientation)
    # Thumb is raised if tip is to the left of the joint (for right hand)
    if (lst.landmark[4].x < lst.landmark[3].x):
        cnt += 1
    
    return cnt  # Return total count of raised fingers

def detect_thumb_gesture(lst):
    """
    Function to detect specific thumb gestures (thumbs up/down)
    lst: MediaPipe hand landmarks object
    """
    thumb_tip_y = lst.landmark[4].y      # Y-coordinate of thumb tip
    thumb_base_y = lst.landmark[3].y     # Y-coordinate of thumb base
    
    # Check thumb position relative to base
    if thumb_tip_y < thumb_base_y - 0.05:  # Thumb significantly above base
        return "thumb_up"
    elif thumb_tip_y > thumb_base_y + 0.05:  # Thumb significantly below base
        return "thumb_down"
    else:
        return "neutral"  # Thumb in neutral position

# Try to initialize camera with different indices
camera_indices = [0, 1, 2, 3, 4, 5]  # Common camera index numbers
cap = None

# Loop through camera indices to find working camera
for index in camera_indices:
    cap = cv2.VideoCapture(index)  # Try to open camera with current index
    if cap.isOpened():  # If camera opens successfully
        print(f"Camera index {index} opened successfully.")
        break  # Exit loop when camera is found
    else:
        cap.release()  # Release camera if it doesn't work

# Exit if no camera is found
if not cap or not cap.isOpened():
    print("Error: Could not open webcam with any of the provided indices.")
    exit()

# Initialize MediaPipe components
drawing = mp.solutions.drawing_utils    # For drawing hand landmarks
hands = mp.solutions.hands              # Hand detection solution
hand_obj = hands.Hands(max_num_hands=1) # Initialize hand detector (max 1 hand)

# Control variables
start_init = False  # Flag to track gesture timing
prev = -1          # Previous gesture state to avoid repeated actions

# Main processing loop
while True:
    end_time = time.time()  # Current time for gesture timing
    ret, frm = cap.read()   # Read frame from camera
    
    # Check if frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break
    
    frm = cv2.flip(frm, 1)  # Flip frame horizontally (mirror effect)
    
    # Process frame for hand detection
    # Convert BGR to RGB (MediaPipe expects RGB)
    res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    
    # If hand landmarks are detected
    if res.multi_hand_landmarks:
        hand_keyPoints = res.multi_hand_landmarks[0]  # Get first hand's landmarks
        
        # Count fingers and detect thumb gesture
        cnt = count_fingers(hand_keyPoints)
        thumb_gesture = detect_thumb_gesture(hand_keyPoints)
        
        print(f"Detected fingers: {cnt}, Thumb gesture: {thumb_gesture}")
        
        # Execute actions only when gesture changes (avoid repeated actions)
        if not(prev == cnt):
            # Initialize timing for gesture stability
            if not(start_init):
                start_time = time.time()
                start_init = True
            # Execute action after 0.2 second delay (gesture stability)
            elif (end_time - start_time) > 0.2:
                
                # Gesture-to-action mapping
                if thumb_gesture == "thumb_up":
                    pyautogui.hotkey('shift', 'n')  # Next video shortcut
                elif thumb_gesture == "thumb_down":
                    pyautogui.hotkey('shift', 'p')  # Previous video shortcut
                elif cnt == 5:  # All fingers raised
                    pyautogui.hotkey('alt', 'left')  # Go back
                elif cnt == 4:  # Four fingers
                    pyautogui.press('right')  # Move right/forward
                elif cnt == 3:  # Three fingers
                    pyautogui.press("down")  # Volume down
                elif cnt == 2:  # Two fingers
                    pyautogui.press("up")  # Volume up
                elif cnt == 1:  # One finger
                    pyautogui.press("space")  # Play/Pause
                elif cnt == 0 and hand_keyPoints.landmark[4].y < hand_keyPoints.landmark[3].y:
                    pyautogui.hotkey('shift', 'n')  # Next video
                elif cnt == 0 and hand_keyPoints.landmark[4].y > hand_keyPoints.landmark[3].y:
                    pyautogui.hotkey('shift', 'p')  # Previous video
                
                prev = cnt  # Update previous gesture
                start_init = False  # Reset timing flag
        
        # Draw hand landmarks on frame for visual feedback
        drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS)
    
    # Display the frame with hand landmarks
    cv2.imshow("window", frm)
    
    # Exit when ESC key is pressed (ASCII 27)
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cv2.destroyAllWindows()  # Close all OpenCV windows
cap.release()            # Release camera resource