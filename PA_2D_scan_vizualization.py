import numpy as np
import matplotlib.pyplot as plt

filename = 'Sample_name-TiN-Full1.npy'

class IndexTracker:
    def __init__(self, ax, data):
        self.ax = ax
        ax.set_title('Photoacoustic signal')
        ax.set_xlabel('us')
        ax.set_ylabel('V')
        self.data = data
        self.x_max = data.shape[1]
        self.y_max = data.shape[0]
        self.x_ind = 0
        self.y_ind = 0

        self.im = ax.plot(self.data[self.x_ind,self.y_ind,:])
        self.update()

    def on_key_press(self, event):

        if event.key == 'left':
            if self.x_ind == 0:
                pass
            else:
                self.x_ind -= 1
        elif event.key == 'right':
            if self.x_ind == (self.x_max - 1):
                pass
            else:
                self.x_ind += 1
        elif event.key == 'down':
            if self.y_ind == 0:
                pass
            else: self.y_ind -= 1
        elif event.key == 'up':
            if self.y_ind == (self.y_max - 1):
                pass
            else:
                self.y_ind += 1
        self.update()


    def update(self):
        self.im.set_data(self.data[self.x_ind,self.y_ind,:])
        title = ('X index = ', self.x_ind, ' Y index = ', self.y_ind)
        self.ax.set_title(title)
        self.im.axes.figure.canvas.draw()

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)


    data = np.load(filename)
    print(data.shape)
    tracker = IndexTracker(ax, data)

    fig.canvas.mpl_connect('key_press_event', tracker.on_key_press)
    plt.show()