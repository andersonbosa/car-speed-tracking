import cv2
import numpy as np

# For YOLO object detection (if using YOLO)
# from darknetpy.darknet import ObjectDetector

# For TensorFlow object detection (if using TensorFlow)
# from tensorflow_object_detector import ObjectDetector


# Initialize the screen capture
def init_screen_capture():
    # screen_capture = cv2.VideoCapture(0)  # Use 0 for the primary screen capture device
    screen_capture = cv2.VideoCapture('/home/t4inha/devspace/the-cheat/_datasets/take4.mp4')
    return screen_capture


object_detector = cv2.createBackgroundSubtractorMOG2()


# Object detection function
def detect_objects(frame):
    mask = object_detector.apply(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 32 and area < 800:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 1)

    # cv2.imshow("Mask", mask)
    return frame


def get_area_of_interest(frame):
    height, width, _ = frame.shape
    print(f"[DEBUG] x:{width}, y:{height}")

    y_1 = int(height / 4)
    y_2 = y_1 + y_1 * 2

    x_1 = 0
    x_2 = int(width)

    roi = frame[y_1:y_2, x_1:x_2]

    return roi


# Main loop
def main():
    screen_capture = init_screen_capture()

    while True:
        ret, frame = screen_capture.read()
        if not ret:
            break

        # Extract area of interest
        roi = get_area_of_interest(frame)

        # Process the frame for object detection
        processed_frame = detect_objects(roi)

        # Display the processed frame (optional, if you want to add GUI rendering)
        cv2.imshow("Object Detection", processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    screen_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
