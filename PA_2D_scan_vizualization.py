import numpy as np
import matplotlib.pyplot as plt

filename = 'measuring results/Sample_name-TiN-Full4.npy'

class IndexTracker:
    def __init__(self, fig, ax, data):
        self.ax = ax
        self.fig = fig
        ax.set_title('Photoacoustic signal')
        ax.set_xlabel('us')
        ax.set_ylabel('V')
        self.data = data
        self.x_max = data.shape[1]
        self.y_max = data.shape[0]
        self.x_ind = 0
        self.y_ind = 0
        self.update()

    def on_key_press(self, event):
        
        if event.key == 'left':
            if self.x_ind == 0:
                pass
            else:
                self.x_ind -= 1
                self.update()
        elif event.key == 'right':
            if self.x_ind == (self.x_max - 1):
                pass
            else:
                self.x_ind += 1
                self.update()
        elif event.key == 'down':
            if self.y_ind == 0:
                pass
            else: 
                self.y_ind -= 1
                self.update()
        elif event.key == 'up':
            if self.y_ind == (self.y_max - 1):
                pass
            else:
                self.y_ind += 1
                self.update()


    def update(self):
        self.ax.clear()
        self.ax.plot(self.data[self.x_ind,self.y_ind,:])
        title = 'X index = ' + str(self.x_ind) + '/' + str(self.x_max-1) + '. Y index = ' + str(self.y_ind) + '/' + str(self.y_max-1)
        self.ax.set_title(title)
        self.fig.canvas.draw()

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)


    data = np.load(filename)
    print(data.shape)
    tracker = IndexTracker(fig, ax, data)

    fig.canvas.mpl_connect('key_press_event', tracker.on_key_press)
    plt.show()