# import the necessary packages
import os
import time
import numpy as np
import imutils
import cv2
import glob
from lineDrawer import LineDrawer
from sort import *

class YoloSortCounter():
    def __init__(self, input_path, output_path, yolo_path, confidence_val, threshold_val):
        # Get inputs from user
        self.input_path = input_path
        self.output_path = output_path
        self.yolo_path = yolo_path
        self.confidence_val = confidence_val
        self.threshold_val = threshold_val

        # Create Sort tracker object
        self.tracker = Sort()

        # Yolo network
        self.layer_net = None
        self.COLORS = None
        self.net = None

        # Main variables
        self.memory = {}
        self.car_counters = []  # One counter for every one line
        self.frame = None  # Current Frame
        self.lines = None  # Lines list

        self.car_IDs_with_line_ids = {}

        # OpenCV Objects
        self.writer = None
        self.vs = None

        # Call main functions
        self.clean_output_dir()
        self.load_yolo()
        self.proccess()
    

    def clean_output_dir(self):
        # Clean output path
        files = glob.glob('output/*')
        for f in files:
            os.remove(f)

    def exit_and_clean(self):
        # release the file pointers
        print("Cleaning up...")
        self.writer.release()
        self.vs.release()
        cv2.destroyAllWindows()

    # Return true if line segments AB and CD intersect
    def intersect(self,A,B,C,D):
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

    def ccw(self,A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def load_yolo(self):
        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([self.yolo_path, "aerial.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of self.COLORS to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(200, 3),
            dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([self.yolo_path, "yolov3-aerial.weights"])
        configPath = os.path.sep.join([self.yolo_path, "yolov3-aerial.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        print("Loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        #self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
       	#self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        ln = self.net.getLayerNames()
        self.layer_net = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def line_draw_callback(self):
        line_drawer_object = LineDrawer(self.frame)
        self.lines = line_drawer_object.drawer()

        self.car_counters = []
        # Create counters for every green line
        for i in range(len(self.lines)):
            if i%2 == 0:
                self.car_counters.append(0)

    def proccess(self):
        # initialize the video stream, pointer to output video file, and
        # frame dimensions
        self.vs = cv2.VideoCapture(self.input_path)
        self.writer = None
        (W, H) = (None, None)

        frameIndex = 0

        # try to determine the total number of frames in the video file
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
            total = int(self.vs.get(prop))
            print("{} total frames in video".format(total))

        # an error occurred while trying to determine the total
        # number of frames in the video file
        except:
            print("Could not determine # of frames in video")
            print("No approximate completion time can be provided")
            total = -1

        # loop over frames from the video file stream
        while True:
            # read the next frame from the file
            (grabbed, self.frame) = self.vs.read()

            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break

            # If lines not draw yet we call linedrawer
            if self.lines is None:
                self.line_draw_callback()

            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = self.frame.shape[:2]

            # construct a blob from the input frame and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes
            # and associated probabilities
            blob = cv2.dnn.blobFromImage(self.frame, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
            self.net.setInput(blob)
            start = time.time()
            layerOutputs = self.net.forward(self.layer_net)
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
                    if confidence > self.confidence_val:
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
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_val, self.threshold_val)
            
            dets = []
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    dets.append([x, y, x+w, y+h, confidences[i]])

            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            dets = np.asarray(dets)
            tracks = self.tracker.update(dets)

            boxes = []
            indexIDs = []
            c = []
            previous = self.memory.copy()
            self.memory = {}

            for track in tracks:
                boxes.append([track[0], track[1], track[2], track[3]])
                indexIDs.append(int(track[4]))
                self.memory[indexIDs[-1]] = boxes[-1]

            # Check founded objects in this frame and add to self.memory if newly appeared
            # -1 is line number and it will cahnge when a green line passed
            for car_id in indexIDs:
                if car_id not in self.car_IDs_with_line_ids:
                    self.car_IDs_with_line_ids[car_id] = -1

            if len(boxes) > 0:
                i = int(0)
                for box in boxes:
                    # extract the bounding box coordinates
                    (x, y) = (int(box[0]), int(box[1]))
                    (w, h) = (int(box[2]), int(box[3]))

                    # draw a bounding box rectangle and label on the image
                    # color = [int(c) for c in self.COLORS[classIDs[i]]]
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    color = [int(c) for c in self.COLORS[indexIDs[i] % len(self.COLORS)]]
                    cv2.rectangle(self.frame, (x, y), (w, h), color, 2)

                    if indexIDs[i] in previous:
                        previous_box = previous[indexIDs[i]]
                        (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                        (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                        p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                        cv2.line(self.frame, p0, p1, color, 3)

                        for indx, l in enumerate(self.lines):
                            if self.intersect(p0, p1, l[0], l[1]):
                                if indx % 2 == 0: # If pass on green
                                    self.car_IDs_with_line_ids[indexIDs[i]] = indx
                                elif indx-1 == self.car_IDs_with_line_ids[indexIDs[i]]: # If pass on red
                                    self.car_counters[(indx-1)//2] += 1

                    # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    text = "{}".format(indexIDs[i])
                    cv2.putText(self.frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    i += 1

            # draw lines
            for indx, l in enumerate(self.lines):
                if indx % 2 == 0:
                    cv2.line(self.frame, l[0], l[1], (0,255,0), 5)
                else:
                    cv2.line(self.frame, l[0], l[1], (0,0,255), 5)
            
            # draw counters
            for i, c in enumerate(self.car_counters):
                (x, y) = self.lines[i*2][0]
                text = "Line "+ str(i)+": " + str(c)
                cv2.putText(self.frame, text, (x - 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

            # Save images to ouput folder
            #cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)
            # Show images
            cv2.imshow("Press q to exit! Press l to draw line!", self.frame)
            # Quit from system
            if cv2.waitKey(25) == ord('q'):
                self.exit_and_clean()
            # Call line drawer    
            elif cv2.waitKey(25) == ord('l'):
                self.line_draw_callback()

            # check if the video writer is None
            if self.writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.writer = cv2.VideoWriter(self.output_path + '/aOUT.avi', fourcc, 30,
                    (self.frame.shape[1], self.frame.shape[0]), True)

                # some information on processing single frame
                if total > 0:
                    elap = (end - start)
                    print("Single frame took {:.4f} seconds".format(elap))
                    print("Estimated total time to finish: {:.4f}".format(
                        elap * total))

            # write the output frame to disk
            self.writer.write(self.frame)

            # increase frame index
            frameIndex += 1

