import cv2
import sys

# Check if the source is passed through command line argument, otherwise default to webcam
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Load the pre-trained model for face detection
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

in_width = 300
in_height = 300
mean = (104.0, 177.0, 123.0)
conf_threshold = 0.7

while cv2.waitKey(1) != 27:  # Escape key to exit
    has_frame, frame = source.read()
    if not has_frame:
        break

    # Flip the frame horizontally (mirror)
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Prepare the frame for deep learning-based face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Iterate over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))

            # Display confidence level
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]), 
                          (x_left_bottom + label_size[0], y_left_bottom + base_line), 
                          (255, 255, 255), cv2.FILLED)

            cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            # Get and display inference time
            t, _ = net.getPerfProfile()
            label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
            cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    # Show the result in the window
    cv2.imshow(win_name, frame)

# Release resources
source.release()
cv2.destroyAllWindows()
