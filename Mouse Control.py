import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Get screen width and height for cursor movement
screen_width, screen_height = pyautogui.size()

# Initialize webcam
camera = cv2.VideoCapture(0)

# Check if the camera is successfully opened
if not camera.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Variables to control the click event interval
last_click_time = 0
click_interval = 0.5  # Minimum time between clicks (in seconds)

# Main loop for hand gesture control
while True:
    ret, image = camera.read()

    if not ret:
        print("Failed to capture image from camera.")
        break

    # Flip the image horizontally for a natural interaction and convert to RGB
    image = cv2.flip(image, 1)
    image_height, image_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    output = hands.process(rgb_image)

    # Initialize status message
    status_message = "Ready"
    click_status = "No Click"

    # Check if any hands are detected
    if output.multi_hand_landmarks:
        for hand_landmarks in output.multi_hand_landmarks:
            # Draw hand landmarks and lines between them
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Track thumb and index finger tips for control
            thumb_tip = None
            index_tip = None

            for id, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)

                # Draw circles at each landmark point for visibility
                if id == 8:  # Index finger tip (move the cursor)
                    index_tip = (x, y)
                    cv2.circle(image, (x, y), 10, (0, 255, 255), -1)
                if id == 4:  # Thumb tip (for click detection)
                    thumb_tip = (x, y)
                    cv2.circle(image, (x, y), 10, (0, 255, 255), -1)

            if index_tip and thumb_tip:
                # Scale the positions to the screen size
                mouse_x = int(screen_width / image_width * index_tip[0])
                mouse_y = int(screen_height / image_height * index_tip[1])

                # Move the mouse cursor
                pyautogui.moveTo(mouse_x, mouse_y)

                # Calculate distance between thumb and index finger tips
                distance = abs(thumb_tip[1] - index_tip[1])

                # Update the status message with the distance
                status_message = f"Distance: {distance:.2f}"

                # Trigger click if the distance between thumb and index finger is small
                if distance < 20 and time.time() - last_click_time > click_interval:
                    pyautogui.click()
                    click_status = "Click Triggered"
                    last_click_time = time.time()  # Update the last click time

    # Display the image with hand landmarks and status
    # Show the status messages on the frame
    cv2.putText(image, f"Status: {status_message}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"Click Status: {click_status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the image with hand landmarks
    cv2.imshow("Hand Gesture Control", image)

    # Exit the program if 'ESC' or 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('q'):  # 'q' key
        print("Exiting the program...")
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
