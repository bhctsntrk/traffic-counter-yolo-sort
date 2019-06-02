# import the necessary packages
import os
import time
import numpy as np
import imutils
import cv2
import glob
from lineDrawer import LineDrawer

# Clean output path
files = glob.glob('output/*')
for f in files:
    os.remove(f)

from sort import *
tracker = Sort()
memory = {}
car_counters = []

car_IDs_with_line_ids = {}

# Get inputs from user
input_path = input("Please enter the input video.\n")
output_path = input("Please enter the output path.\n")
yolo_path = "./yolo-coco"
confidence_val = 0.5
threshold_val = 0.3

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([yolo_path, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(input_path)
writer = None
(W, H) = (None, None)

frameIndex = 0

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("{} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("Could not determine # of frames in video")
    print("No approximate completion time can be provided")
    total = -1

_, first_frame = vs.read()
line_drawer_object = LineDrawer(first_frame)
lines = line_drawer_object.drawer()

# Create counters for every green line
for i in range(len(lines)):
    if i%2 == 0:
        car_counters.append(0)

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_val:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_val, threshold_val)
    
    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]])

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    # Check founded objects in this frame and add to memory if newly appeared
    # -1 is line number and it will cahnge when a green line passed
    for car_id in indexIDs:
        if car_id not in car_IDs_with_line_ids:
            car_IDs_with_line_ids[car_id] = -1

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                cv2.line(frame, p0, p1, color, 3)

                for indx, l in enumerate(lines):
                    if intersect(p0, p1, l[0], l[1]):
                        if indx % 2 == 0: # If pass on green
                            car_IDs_with_line_ids[indexIDs[i]] = indx
                        elif indx-1 == car_IDs_with_line_ids[indexIDs[i]]: # If pass on red
                            car_counters[(indx-1)//2] += 1

            # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            text = "{}".format(indexIDs[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    # draw lines
    for indx, l in enumerate(lines):
        if indx % 2 == 0:
            cv2.line(frame, l[0], l[1], (0,255,0), 5)
        else:
            cv2.line(frame, l[0], l[1], (0,0,255), 5)
    
    # draw counters
    for i, c in enumerate(car_counters):
        (x, y) = lines[i*2][0]
        text = "Line "+ str(i)+": " + str(c)
        cv2.putText(frame, text, (x - 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

	# Save images to ouput folder
    #cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)
    # Show images
    cv2.imshow("Main Screen Press q to exit!", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
    	break

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path + '/aOUT.avi', fourcc, 30,
            (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("Single frame took {:.4f} seconds".format(elap))
            print("Estimated total time to finish: {:.4f}".format(
                elap * total))

    # write the output frame to disk
    writer.write(frame)

    # increase frame index
    frameIndex += 1
    
# release the file pointers
print("Cleaning up...")
writer.release()
vs.release()
cv2.destroyAllWindows()
