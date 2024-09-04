
import cv2
import numpy as np


#Reads the video using VideoCapture
cap = cv2.VideoCapture('Video3.mp4')


#This retrieves the width and height of the frames captured by the video source 
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

#Initializing VideoWrite format
out = cv2.VideoWriter("output.mp4", fourcc, 5.0, (1280,720))
#Parameter - output.mp4 - Name of the output file
#            fourcc - This is the FourCC representing the codec to be used for encoding the video
#            5.0 - Frame rate at which the video will be saved
#            (1280,720) - width and height of video frame

# Reading 2 frames, one after another 
ret, frame1 = cap.read()
ret, frame2 = cap.read()

#While loop continues
while cap.isOpened():

    #difference between 2 frames captured. abssdiff - absolute difference
    diff = cv2.absdiff(frame1, frame2)

    #Coverts diff image to grayscale image 
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    #This applies Gaussian blurring to the grayscale difference image to reduce noise and smooth out small details.
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #Parameters - gray - source of image
    #             (5,5) - Kernal size/ k-size - This tuple specifies the size of the kernel used for blurring. In this case, it's a 5x5 kernel. 
    #                                           The larger the kernel size, the more significant the blurring effect.
    #              0 - This parameter specifies the standard deviation of the Gaussian kernel in the X direction.
   
   #This line of code performs thresholding on the blurred image 'blur'.
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    #Parameters - blur - Img src
    #             20 - thresh value
    #             255 - maximum thresh value
    #             cv2.THRESH_BINARY - This is a flag indicating the type of thresholding to perform.
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_DUPLEX,
                    1, (0, 0, 255), 3)
        
    M = cv2.moments (c)
    cx = int (M["m10"]/ M["m00"])
    cy = int(M["m01"]/M["m00"])
    cv2.circle (thresh, (cx,cy),7, (255,255,255), -1)
    cv2.putText(thresh, "Centre", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    image = cv2.resize(frame1, (1280,720))
    out.write(image)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()
