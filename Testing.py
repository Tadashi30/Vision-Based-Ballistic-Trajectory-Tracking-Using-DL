import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

# List to store centroid coordinates
centroid_path = []

while cap.isOpened():
    if frame1 is None or frame2 is None:
        break

    # Resize frames if they don't match in size
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

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

cap.release()
cv2.destroyAllWindows()

# Apply K-Means clustering on the collected centroids
if centroid_path:
    X = np.array(centroid_path)
    kmeans = KMeans(n_clusters=2, random_state=0)  # Set n_clusters to your desired number of clusters
    kmeans.fit(X)

    # Get the labels and cluster centers
    labels = kmeans.predict(X)
    centers = kmeans.cluster_centers_

    # Plot the clustered centroids
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', label='Centers')
    plt.title('K-Means Clustering of Centroid Path')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.legend()
    plt.show()
