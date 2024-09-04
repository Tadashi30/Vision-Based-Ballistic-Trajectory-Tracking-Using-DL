import cv2
import numpy as np

# Reads the video using VideoCapture
cap = cv2.VideoCapture('Pencil6.mp4')

# Retrieves the width and height of the frames captured by the video source 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

# Initializing VideoWrite format
out = cv2.VideoWriter("output.mp4", fourcc, 5.0, (frame_width, frame_height))

# Reading 2 frames, one after another 
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# While loop continues
while cap.isOpened():

    # Difference between 2 frames captured. absdiff - absolute difference
    diff = cv2.absdiff(frame1, frame2)

    # Coverts diff image to grayscale image 
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # This applies Gaussian blurring to the grayscale difference image to reduce noise and smooth out small details.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # This line of code performs thresholding on the blurred image 'blur'.
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through each contour
    for contour in contours:
        # Calculate moments of the contour
        M = cv2.moments(contour)
        # Calculate centroid coordinates
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Draw centroid on frame1
            cv2.circle(frame1, (cX, cY), 5, (0, 0, 255), -1)
    
    # This line displays the original frame1 in a window titled "feed". This allows you to see the video feed with bounding boxes and centroids drawn on it.
    cv2.imshow("feed", frame1)

    # This line updates frame1 to be equal to frame2. This prepares frame1 for the next iteration of the loop, where frame2 will be read from the video.
    frame1 = frame2

    # This line reads the next frame from the video (cap) and assigns it to frame2. The ret variable indicates whether the frame was successfully read (True or False).
    ret, frame2 = cap.read()

    # This line checks if the user pressed the 'Esc' key (ASCII code 27). If the 'Esc' key is pressed, the condition becomes True, and the loop breaks, exiting the program.
    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()
