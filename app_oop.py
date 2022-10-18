import tkinter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed


class DrawFig:

    def __init__(self):
        self.root = tkinter.Tk()
        self.root.wm_title("Embedding in Tk")
        # Set Window to Full Screen
        self.root.state('zoomed')
        # self.root.attributes('-fullscreen', True)
        # self.screen_width = self.root.winfo_screenwidth()
        # self.screen_height = self.root.winfo_screenheight()
        # self.root.geometry("%dx%d" % (self.screen_width, self.screen_height))

        # self.fig = Figure()
        self.fig = plt.figure()
        self.t = np.arange(0, 3, .01)
        self.ax = self.fig.add_subplot(111)
        self.a = 2
        self.f = 2
        self.id = 0
        self.data_array = np.array([
            self.a * np.sin(self.f * np.pi * self.t),
            self.a * 4 * np.sin(self.f * np.pi * self.t),
            self.a * 2 * np.sin(self.f * 4 * np.pi * self.t)])

        # self.data, = self.ax.plot(self.t, self.a * np.sin(self.f * np.pi * self.t))
        self.data, = self.ax.plot(self.t, self.data_array[self.id])

        self.frame = tkinter.Frame(master=self.root)
        self.frame.pack(side=tkinter.TOP)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        # side: align to top (put it below the other)
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.redbutton = tkinter.Button(self.frame, text="Red", fg="red")
        self.redbutton.pack(side=tkinter.LEFT)

        self.button = tkinter.Button(master=self.root, text="Quit", command=self._quit)
        self.button.pack(side=tkinter.BOTTOM)

        self.button_1 = tkinter.Button(master=self.root, text="Increase", command=self._increase_freq)
        self.button_1.pack(side=tkinter.LEFT)

        self.button_2 = tkinter.Button(master=self.root, text="Decrease", command=self._decrease_freq)
        self.button_2.pack(side=tkinter.RIGHT)

        self.canvas.mpl_connect("key_press_event", self.on_key_press)

    def on_key_press(self, event):
        if event.key == 'left':
            # Decrease Index by 1
            self.id -= 1
            self.id = self.id % self.data_array.shape[0]
            self.data.set_ydata(self.data_array[self.id])
            self.redbutton['text'] = 'This was LEFT'
            self.redbutton['bg'] = 'green'
            self.canvas.draw()
        elif event.key == 'right':
            self.id += 1
            self.id = self.id % self.data_array.shape[0]
            self.data.set_ydata(self.data_array[self.id])
            self.redbutton['text'] = 'This was RIGHT'
            self.redbutton['bg'] = 'black'

            self.canvas.draw()
        print("you pressed {}".format(event.key))
        key_press_handler(event, self.canvas, self.toolbar)

    def _quit(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
                             # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    def _increase_freq(self):
        self.f = 2 * self.f
        self.data.set_ydata(self.a * np.sin(self.f * np.pi * self.t))
        self.canvas.draw()

    def _decrease_freq(self):
        self.f = 0.5 * self.f
        self.data.set_ydata(self.a * np.sin(self.f * np.pi * self.t))
        self.canvas.draw()


if __name__ == "__main__":
    app = DrawFig()
    tkinter.mainloop()
    # If you put root.destroy() here, it will cause an error if the window is
    # closed with the window manager.()