import cv2
import numpy as np

# Reads the video using VideoCapture
cap = cv2.VideoCapture('Pencil6.mp4')

# Retrieves the width and height of the frames captured by the video source 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

# Initializing VideoWrite format
out = cv2.VideoWriter("output.mp4", fourcc, 5.0, (1280, 720))

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

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue
        
        # Calculate centroid of the rectangle
        centroid_x = x + w // 2
        centroid_y = y + h // 2

        # Draw bounding rectangle and centroid
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame1, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # This line resizes the frame1 to a resolution of 1280x720 pixels and assigns it to the variable 'image'. This step ensures that all frames have the same size for writing to the output video.
    image = cv2.resize(frame1, (1280, 720))

    # This line writes the resized frame (image) to the output video file (out). The write() function adds the frame to the video file.
    out.write(image)

    # This line displays the original frame1 in a window titled "feed". This allows you to see the video feed with bounding boxes and centroids drawn on it.
    cv2.imshow("feed", frame1)

    # This line updates frame1 to be equal to frame2. This prepares frame1 for the next iteration of the loop, where frame2 will be read from the video.
    frame1 = frame2

    # This line reads the next frame from the video (cap) and assigns it to frame2. The ret variable indicates whether the frame was successfully read (True or False).
    ret, frame2 = cap.read()

    # This line checks if the user pressed the 'Esc' key (ASCII code 27). If the 'Esc' key is pressed, the condition becomes True, and the loop breaks, exiting the program.
    if cv2.waitKey(100) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()
