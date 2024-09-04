import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture
cap = cv2.VideoCapture('Pencil6.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a larger figure for the plot
plt.figure(figsize=(5, 5))

# Read initial frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# List to store centroid coordinates and features
centroid_path = []
features = []
labels = []

# Prepare the SVM classifier
clf = svm.SVC(kernel='linear', probability=True)
scaler = StandardScaler()

# Initialize the HOG descriptor
hog = cv2.HOGDescriptor(
    _winSize=(32, 32),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

while cap.isOpened():
    # Calculate absolute difference between consecutive frames
    diff = cv2.absdiff(frame1, frame2)

    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding and dilation
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour and its centroid
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            centroid_path.append((centroid_x, centroid_y))

            # Extract features for SVM
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Check if ROI is within the bounds of the frame
            if x >= 0 and y >= 0 and (x + w) <= frame1.shape[1] and (y + h) <= frame1.shape[0]:
                if w > 5 and h > 5:  # Skip very small ROIs
                    roi = frame1[y:y+h, x:x+w]
                    resized_roi = cv2.resize(roi, (32, 32))  # Reduce size to 32x32
                    
                    if resized_roi.size != 0 and resized_roi.shape[:2] == (32, 32):  # Ensure ROI is not empty and is 32x32
                        try:
                            feature = hog.compute(resized_roi).flatten()
                            label = 1  # Dummy label for now, replace with actual label
                            features.append(feature)
                            labels.append(label)

                            # Draw centroid and path
                            cv2.circle(frame1, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
                            if len(centroid_path) > 1:
                                for i in range(1, len(centroid_path)):
                                    cv2.line(frame1, centroid_path[i - 1], centroid_path[i], (0, 255, 0), 8)

                            # Plot centroid trajectory
                            x_coords, y_coords = zip(*centroid_path)
                            plt.clf()
                            plt.plot(x_coords, y_coords)
                            plt.title('Centroid Trajectory')
                            plt.xlabel('X')
                            plt.ylabel('Y')
                            plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
                            plt.draw()
                            plt.pause(0.01)
                        except cv2.error as e:
                            print("Error computing HOG features:", e)
                            continue  # Skip this ROI if there's an error
                    else:
                        print("Skipping invalid or empty ROI of shape:", resized_roi.shape)
                        continue

    # Calculate and draw progress bar
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    progress_percent = (frame_number / total_frames) * 100
    timeline_length = int(progress_percent * frame_width / 100)
    cv2.rectangle(frame1, (0, frame_height - 10), (timeline_length, frame_height), (255, 255, 255), -1)

    # Overlay timestamp
    current_time = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    minutes, seconds = divmod(current_time, 60)
    timestamp = f'{minutes:02d}:{seconds:02d}'
    cv2.putText(frame1, timestamp, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("feed", frame1)

    # Exit on 'Esc' key press
    if cv2.waitKey(50) == 27:
        break

    # Update frames
    frame1, frame2 = frame2, cap.read()[1]

# Train SVM with collected features
if features and labels:
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf.fit(X_train, y_train)
    print("SVM trained with accuracy:", clf.score(X_test, y_test))

cap.release()
cv2.destroyAllWindows()
plt.close('all')
