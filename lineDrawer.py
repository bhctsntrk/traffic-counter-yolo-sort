import cv2
import copy

class LineDrawer():
    def __init__(self, frame, black_border_height):
        self.drawing = False
        self.lines = []
        self.p1 = None
        self.p2 = None
        self.frame = frame

        self.title = "Draw Lines"

        self.black_border_height = black_border_height

    def on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.p1 = (x,y-self.black_border_height)  # Because black area is in upper section of image and 

        elif event == cv2.EVENT_LBUTTONUP:
            self.p2 = (x,y-self.black_border_height)  # we need to substract height from y values of lines 
            self.lines.append([self.p1, self.p2])


        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing is True:
                self.p2 = (x,y)

    def show_output(self, frame):
        # Show image with warnings
        
        b_frame = cv2.copyMakeBorder(self.frame,self.black_border_height,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])

        text = ["Please draw at least two lines in frame",
                "Press r key to reset.",
                "Press o key when you finish it."]

        for indx,i in enumerate(text):
            cv2.putText(b_frame, i, (40, 30*(indx+1)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, [255,255,255], 2)

        cv2.imshow(self.title, b_frame)

    def drawer(self):
        print("Please draw lines in frame")
        print("Press r key to reset.")
        print("Press o key when you finish it.")

        # We need deep copy to copy original frame
        frame_pure_copy = copy.deepcopy(self.frame)

        cv2.namedWindow(self.title)
        cv2.setMouseCallback(self.title, self.on_mouse)

        while True:
            
            if self.lines:
                for indx, l in enumerate(self.lines):
                        cv2.line(self.frame, l[0], l[1], (0,255,0), 5)

            self.show_output(self.frame)

            # We need to do this because event handler too fast and too many border added to image
            # So we reset image every iteration
            self.frame = frame_pure_copy
            
            key = cv2.waitKey(1)
            if key == ord('o'):
                break
            elif key == ord('r'):
                self.lines = []
                self.frame = frame_pure_copy

        cv2.destroyAllWindows()

        return self.lines
