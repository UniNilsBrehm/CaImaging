import tkinter
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import os
import analysis_util_functions as uf
import pandas as pd


def import_f_raw(file_dir, change_header):
    if change_header:
        data = pd.read_csv(file_dir, decimal='.', sep='\t', header=None).drop(columns=0)
    else:
        data = pd.read_csv(file_dir, decimal='.', sep='\t', header=0, index_col=0)
    header_labels = []
    for kk in range(data.shape[1]):
        header_labels.append(f'roi_{kk + 1}')
    data.columns = header_labels
    return data


class DrawFig:

    def __init__(self, data, stimulus, fr_data, fr_stimulus):
        # Get Input Variables
        self.fr_stimulus = 1000
        self.data = data / np.max(data, axis=0)
        self.stimulus = stimulus
        self.fr_data = fr_data
        self.fr_stimulus = fr_stimulus
        self.rois = self.data.keys()
        self.id = 0
        # Convert Samples to Time
        self.time_data = np.linspace(0, len(self.data) / self.fr_data, len(self.data))
        self.time_stimulus = np.linspace(0, len(self.stimulus) / self.fr_stimulus, len(self.stimulus))

        self.root = tkinter.Tk()
        self.root.wm_title("SIMPLE DATA VIEWER")
        # Set Window to Full Screen
        self.root.state('zoomed')

        # FIGURE START
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.data_to_plot, = self.ax.plot(self.time_data, self.data[self.rois[self.id]], 'k')
        self.stimulus_to_plot, = self.ax.plot(self.time_stimulus, self.stimulus, 'b')
        self.ax.set_title(f'roi: {self.rois[self.id]}')
        # FIGURE END

        # Add Menu
        self.menu_bar = tkinter.Menu(self.root)
        self.file_menu = tkinter.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Open", command=self._open_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self._quit)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.root.config(menu=self.menu_bar)

        self.frame = tkinter.Frame(master=self.root)
        self.frame.pack(side=tkinter.TOP)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        # side: align to top (put it below the other)
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # Add Buttons
        self.redbutton = tkinter.Button(self.frame, text="Red", fg="red")
        self.redbutton.pack(side=tkinter.LEFT)

        self.button = tkinter.Button(master=self.root, text="Quit", command=self._quit)
        self.button.pack(side=tkinter.BOTTOM)

        self.button_1 = tkinter.Button(master=self.root, text="PREVIOUS", command=self._previous)
        self.button_1.pack(side=tkinter.LEFT)

        self.button_2 = tkinter.Button(master=self.root, text="NEXT", command=self._next)
        self.button_2.pack(side=tkinter.RIGHT)

        self.canvas.mpl_connect("key_press_event", self.on_key_press)

    def _open_file(self):
        # file_types example: [('Recording Files', 'raw.txt')]
        self.file_name = filedialog.askopenfilename(filetypes=[('Recording Files', 'raw.txt')])
        self.data_file_name = os.path.split(self.file_name)[1]
        self.rec_dir = os.path.split(self.file_name)[0]
        self.rec_name = os.path.split(self.rec_dir)[1]
        # Import Data
        self.data = import_f_raw(f'{self.rec_dir}/{self.rec_name}_raw.txt', change_header=False)
        # Import Stimulus
        self.stimulus = uf.import_txt_stimulation_file(data_path=self.rec_dir, data_name=f'{self.rec_name}_stimulation.txt')
        self.fr_rec = uf.estimate_sampling_rate(data=self.data, f_stimulation=self.stimulus, print_msg=False)

    def on_key_press(self, event):
        if event.key == 'left':
            # Decrease Index by 1
            self._previous()
            # self.redbutton['text'] = 'This was LEFT'
            # self.redbutton['bg'] = 'green'
        elif event.key == 'right':
            self._next()
            # self.redbutton['text'] = 'This was RIGHT'
            # self.redbutton['bg'] = 'black'

        elif event.key == 'q':
            self._quit()

        # print("you pressed {}".format(event.key))
        key_press_handler(event, self.canvas, self.toolbar)

    def _quit(self):
        self.root.quit()     # stops mainloop
        # this is necessary on Windows to prevent Fatal Python Error: PyEval_RestoreThread: NULL state
        self.root.destroy()

    def _next(self):
        self.id += 1
        self.id = self.id % self.data.shape[1]
        self._turn_page()

    def _previous(self):
        self.id -= 1
        self.id = self.id % self.data.shape[1]
        self._turn_page()

    def _turn_page(self):
        self.data_to_plot.set_ydata(self.data[self.rois[self.id]])
        self.ax.set_title(f'roi: {self.rois[self.id]}')
        self.canvas.draw()


if __name__ == "__main__":
    # Select Directory and get all files
    file_name = uf.select_file([('Recording Files', 'raw.txt')])
    data_file_name = os.path.split(file_name)[1]
    rec_dir = os.path.split(file_name)[0]
    rec_name = os.path.split(rec_dir)[1]
    uf.msg_box(rec_name, f'SELECTED RECORDING: {rec_name}', '+')

    # Get Raw Values
    stimulus = uf.import_txt_stimulation_file(data_path=rec_dir, data_name=f'{rec_name}_stimulation.txt')
    f_raw = import_f_raw(f'{rec_dir}/{rec_name}_raw.txt', change_header=False)
    fr_rec = uf.estimate_sampling_rate(data=f_raw, f_stimulation=stimulus, print_msg=False)
    # Get Stimulus
    app = DrawFig(data=f_raw, stimulus=(stimulus['Volt'] * -1) / np.max(stimulus['Volt'] * -1), fr_data=fr_rec, fr_stimulus=1000)
    tkinter.mainloop()
    # If you put root.destroy() here, it will cause an error if the window is
    # closed with the window manager.()
