import numpy as np
import matplotlib.pyplot as plt
import configparser

config = configparser.ConfigParser()
config.read('SimulationParams.ini')

model_area_x = int(config['Model Geometry']['model area x [mm]'])
model_area_y = int(config['Model Geometry']['model area y [mm]'])
model_area_z = int(config['Model Geometry']['model area z [mm]'])
data_x_size = int(config['Model Geometry']['data x size'])
data_y_size = int(config['Model Geometry']['data y size'])
data_z_size = int(config['Model Geometry']['data z size'])
simulation_filename = config['Technical data']['absorbed light filename']

class IndexTracker:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')
        ax.set_xlabel('mm')
        ax.set_xticks([i*(data_x_size-1)/5 for i in range(6)])
        ax.set_xticklabels([i*model_area_x/5 for i in range(6)])
        ax.set_ylabel('mm')
        ax.set_yticks([i*(data_z_size-1)/5 for i in range(6)])
        ax.set_yticklabels([i*model_area_z/5 for i in range(6)])
        self.X = X
        self.slices = X.shape[1]
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, self.ind, :], 
                            vmin = 0, vmax = np.amax(self.X))
        self.cbar = plt.colorbar(self.im, ax=self.ax)
        self.cbar.set_label('absorbed photons', rotation=90)
        self.update()

    def on_scroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1)
            if self.ind >= self.slices:
               self.ind = (self.ind - 1)
        else:
            self.ind = (self.ind - 1)
            if self.ind < 0:
                self.ind = (self.ind + 1)
        self.update()

    def update(self):
        self.im.set_data(self.X[:, self.ind, :])
        coordinate_y = '%.2f' % (self.ind/data_y_size*model_area_y)
        title = ('Y coordinate ' + coordinate_y +' mm out of ' + str(model_area_y) + ' mm')
        self.ax.set_title(title)
        self.im.axes.figure.canvas.draw()

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)

    X = np.load(simulation_filename)
    print(X.shape)
    tracker = IndexTracker(ax, X)

    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()