import tkinter as tk
from tkinter import filedialog as fd
from yolo import YoloSortCounter
import sys

class TkWindow(tk.Frame):
    # We need tk.Var series because they updated in real time
    def __init__(self, master, initialdir='', filetypes=()):
        super().__init__(master)
        
        self._initaldir = initialdir
        self._filetypes = filetypes
        self.input_filepath = tk.StringVar()
        self.output_path = tk.StringVar()
        self.yolo_path = tk.StringVar()

        self.confidence = tk.IntVar()
        self.threshold = tk.IntVar()        

        self._create_widgets()
        self._display_widgets()

    def _create_widgets(self):
        self._input_label = tk.Label(self,text="Input File")
        self._input_entry = tk.Entry(self, textvariable=self.input_filepath)
        self._input_button = tk.Button(self, text="Browse...", command=self.input_browse)

        self._output_label = tk.Label(self,text="Output Folder")
        self._output_entry = tk.Entry(self, textvariable=self.output_path)
        self._output_button = tk.Button(self, text="Browse...", command=self.output_browse)

        self._yolo_label = tk.Label(self,text="Yolo Folder")
        self._yolo_entry = tk.Entry(self, textvariable=self.yolo_path)
        self._yolo_button = tk.Button(self, text="Browse...", command=self.yolo_browse)

        self._confidence_label = tk.Label(self,text="Confidence Value")
        self._confidence_slider = tk.Scale(self, from_=0.0, to=1.0, length=300, resolution=0.01, orient=tk.HORIZONTAL)
        self._confidence_slider.set(0.5)

        self._threshold_label = tk.Label(self,text="Threshold Value")
        self._threshold_slider = tk.Scale(self, from_=0.0, to=1.0, length=300, resolution=0.01, orient=tk.HORIZONTAL)
        self._threshold_slider.set(0.3)

        self._start_button = tk.Button(self, text="Start!", command=self.start_counter)

        self._stop_button = tk.Button(self, text="Stop!", command=self.stop_counter)

    def _display_widgets(self):
        self._input_label.pack()
        self._input_entry.pack(fill='x', expand=True)
        self._input_button.pack(anchor='se')

        self._output_label.pack()
        self._output_entry.pack(fill='x', expand=True)
        self._output_button.pack(anchor='se')

        self._yolo_label.pack()
        self._yolo_entry.pack(fill='x', expand=True)
        self._yolo_button.pack(anchor='se')

        self._confidence_label.pack()
        self._confidence_slider.pack()

        self._threshold_label.pack()
        self._threshold_slider.pack()

        self._start_button.pack(anchor='se')

        self._stop_button.pack(anchor='se')

    def input_browse(self):
        self.input_filepath.set(fd.askopenfilename(initialdir=self._initaldir, filetypes=self._filetypes))
    def output_browse(self):
        self.output_path.set(fd.askdirectory(initialdir=self._initaldir))
    def yolo_browse(self):
        self.yolo_path.set(fd.askdirectory(initialdir=self._initaldir))

    def start_counter(self):
        yoloSortCounter = YoloSortCounter(self.input_filepath.get(), self.output_path.get(), self.yolo_path.get(), self.confidence.get(), self.threshold.get())
    def stop_counter(self):
        sys.exit()

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Yolo Solo Counter Control Panel')
    root.geometry("550x420")

    tkObject = TkWindow(root, initialdir=r".", filetypes=(('Moving Picture Experts Group 4','*.mp4'), ('Audio Video Interleave','*.avi'), ("All files", "*.*")))
    tkObject.pack(fill='x', expand=True)

    root.mainloop()