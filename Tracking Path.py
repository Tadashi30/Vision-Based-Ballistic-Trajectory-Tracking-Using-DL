import cv2
import numpy as np

# Reads the video using VideoCapture
cap = cv2.VideoCapture('Pencil6.mp4')

# Retrieves the width and height of the frames captured by the video source 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Reading 2 frames, one after another 
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# List to store centroid coordinates
centroid_path = []

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
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate the centroid of the largest contour
            M = cv2.moments(largest_contour)
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            # Append centroid coordinates to the centroid_path list
            centroid_path.append((centroid_x, centroid_y))
            # Draw the centroid on the resized image
            cv2.circle(frame1, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
            cv2.putText(frame1, "centroid", (centroid_x - 25, centroid_y - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Draw path of centroid
            for i in range(1, len(centroid_path)):
             cv2.line(frame1, centroid_path[i - 1], centroid_path[i], (0, 255, 0), 8)
             
                        # Display the centroid coordinates
            coords_text = f'({centroid_x}, {centroid_y})'
            cv2.putText(frame1, coords_text, (centroid_x + 10, centroid_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Calculate the percentage of completion of the video
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_percent = (frame_number / total_frames) * 100

    # Draw a timeline bar on the frame to represent the progress
    timeline_length = int(progress_percent * frame_width / 100)
    cv2.rectangle(frame1, (0, frame_height - 10), (timeline_length, frame_height), (255, 255, 255), -1)

    # Calculate the current timestamp of the video
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_percent = (current_frame / total_frames) * 100
    current_time = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    minutes = current_time // 60
    seconds = current_time % 60

    # Overlay the timestamp onto the frame
    timestamp = f'{minutes:02d}:{seconds:02d}'
    text_height = int(cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1])
    cv2.putText(frame1, timestamp, (10, frame_height - 10 - text_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # This line displays the original frame1 in a window titled "feed". This allows you to see the video feed with bounding boxes and centroids drawn on it.
    cv2.imshow("feed", frame1)

    # This line updates frame1 to be equal to frame2. This prepares frame1 for the next iteration of the loop, where frame2 will be read from the video.
    frame1 = frame2

    # This line reads the next frame from the video (cap) and assigns it to frame2. The ret variable indicates whether the frame was successfully read (True or False).
    ret, frame2 = cap.read()

    # This line checks if the user pressed the 'Esc' key (ASCII code 27). If the 'Esc' key is pressed, the condition becomes True, and the loop breaks, exiting the program.
    key = cv2.waitKey(80)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(0) #wait until any key is pressed

    # Draw path of centroid on frame1 before saving the snapshot
        for i in range(1, len(centroid_path)):
            cv2.line(frame1, centroid_path[i - 1], centroid_path[i], (0, 255, 0), 8)
            cv2.circle(frame1, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
            cv2.putText(frame1, "centroid", (centroid_x - 25, centroid_y - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
cap.release()
cv2.destroyAllWindows()

