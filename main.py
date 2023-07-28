import os
from pathlib import Path

import cv2
import numpy as np
from tracker import EuclideanDistTracker

main_dir_path = os.path.dirname(__file__)

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
tracker = EuclideanDistTracker()


# Initialize the screen capture
def init_screen_capture():
    # screen_capture = cv2.VideoCapture(0)  # Use 0 for the primary screen capture device
    screen_capture = cv2.VideoCapture(
        str(Path(f"{main_dir_path}/assets/video1.mp4").resolve())
    )
    return screen_capture


# Object detection function
def process_frame(frame):
    # 1. Object detection
    mask = object_detector.apply(frame)

    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 1)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            detections.append([x, y, w, h])

    # 2. Object tracking
    print(f"[DEBUG] {detections}")
    boxes_id = tracker.update(detections)
    for box_id in boxes_id:
        x, y, w, h, _id = box_id
        cv2.putText(
            frame, str(_id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.waitKey(0) # helps debug frame per
    cv2.imshow("Mask", mask)
    return frame


def get_area_of_interest(frame):
    height, width, _ = frame.shape
    # print(f"[DEBUG] x:{width}, y:{height}")

    y_1 = int(height / 4)
    y_2 = y_1 * 3

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
        _processed_frame = process_frame(roi)

        # Display the processed frame (optional, if you want to add GUI rendering)
        cv2.imshow("Frame", frame)
        # cv2.imshow("Processed Frame", processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    screen_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
