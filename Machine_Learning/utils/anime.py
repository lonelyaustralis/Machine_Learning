import matplotlib.pyplot as plt
import numpy as np


class Anime_plot:
    def __init__(self, x_lim: list, y_lim: list, data_x_start_point: list, data_y_start_point: list, legend: list):
        assert len(x_lim) == 2 and len(y_lim) == 2, "x_lim and y_lim must be list of length 2"
        assert (
            len(data_x_start_point) == len(data_y_start_point) == len(legend)
        ), "data_x_start_point, data_y_start_point and legend must be same length"
        self.fig, self.ax = plt.subplots()
        self.legend = legend
        self.plot_list = []
        self.plot_xdata = []
        self.plot_ydata = []
        for i in range(len(data_x_start_point)):
            self.plot_list.append(self.ax.plot(data_x_start_point[i], data_y_start_point[i], animated=True, label=legend[i]))
            self.plot_xdata.append(np.array([data_x_start_point[i]]))
            self.plot_ydata.append(np.array([data_y_start_point[i]]))
        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)
        self.ax.legend(loc=4)
        self.ax.grid()
        plt.show(block=False)
        plt.pause(0.1)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for i in range(len(self.plot_list)):
            self.ax.draw_artist(self.plot_list[i][0])
        self.fig.canvas.blit(self.fig.bbox)

    def update(self, data_y_update: list, data_x_update: list = None):
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        if data_x_update is not None:
            assert len(data_x_update) == len(data_y_update), "data_x_update and data_y_update must be same length"
            for i in range(len(data_y_update)):
                self.plot_xdata[i] = np.append(self.plot_xdata[i], data_x_update[i])
        else:
            for i in range(len(data_y_update)):
                self.plot_xdata[i] = np.append(self.plot_xdata[i], self.plot_xdata[i][-1] + 1)
        for i in range(len(data_y_update)):
            self.plot_ydata[i] = np.append(self.plot_ydata[i], data_y_update[i])
        self.fig.canvas.restore_region(self.bg)
        for i in range(len(self.plot_list)):
            self.plot_list[i][0].set_data(self.plot_xdata[i], self.plot_ydata[i])
            self.ax.draw_artist(self.plot_list[i][0])
        self.fig.canvas.blit(self.fig.bbox)
        # flush any pending GUI events, re-painting the screen if needed
        self.fig.canvas.flush_events()

    def end(self):
        self.ax.cla()
        for i in range(len(self.plot_list)):
            self.ax.plot(self.plot_xdata[i], self.plot_ydata[i], label=self.plot_list[i][0].get_label())
        self.ax.legend(loc=4)
        self.ax.grid()
        plt.show()

