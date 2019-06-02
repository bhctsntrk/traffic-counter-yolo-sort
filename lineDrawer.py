import cv2
import copy

class LineDrawer():
    def __init__(self, frame):
        self.drawing = False
        self.lines = []
        self.p1 = None
        self.p2 = None
        self.frame = frame

    def on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing is False:
                self.drawing = True
                self.p1 = (x,y)
            else:
                self.drawing = False
                self.lines.append([self.p1, self.p2])


        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing is True:
                self.p2 = (x,y)

    def drawer(self):
        print("Please draw lines in frame")
        print("Press r key to reset.")
        print("Press o key when you finish it.")

        # We need deep copy to copy original frame
        frame_pure_copy = copy.deepcopy(self.frame)

        cv2.namedWindow('Draw Lines')
        cv2.setMouseCallback('Draw Lines', self.on_mouse)

        while True:
            
            if self.lines:
                for indx, l in enumerate(self.lines):
                    if indx % 2 == 0:
                        cv2.line(self.frame, l[0], l[1], (0,255,0), 5)
                    else:
                        cv2.line(self.frame, l[0], l[1], (0,0,255), 5)

            cv2.imshow('Draw Lines', self.frame)
            
            key = cv2.waitKey(1)
            if key == ord('o'):
                break
            elif key == ord('r'):
                self.lines = []
                self.frame = frame_pure_copy

        cv2.destroyAllWindows()

        return self.lines
