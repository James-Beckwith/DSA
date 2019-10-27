import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage, imsave
from matplotlib.path import Path
from PIL import Image

'''Class to pick and store masks on an image'''
class imagePicking:

    def extents(self, f):
      delta = f[1] - f[0]
      return [f[0] - delta/2, f[-1] + delta/2]

    def __init__(self, inputData):
        ''' Initialization.
            inputData - 2D numpy array representing image to pick on.
                        shape of inputData = [# Y sample, # of X samples].'''

        # store size of input data
        self.numX, self.numY = np.shape(inputData)
        self.inputData = inputData


    ''' function to pick a window based on the FK domain response of the input data '''
    def pickWindow(self, name):

        # plot FK amplitude reponse for positive frequencies only
        fig, ax = plt.subplots()
        plt.imshow(self.inputData, aspect='auto', interpolation='none', picker=True)

        picks = []
        # define what to do when a pick is made, mouse button 1-> store picks,
        # mouse button 3->stop picking
        def onpick(event):
            if event.button == 1:
                picks.append([event.xdata, event.ydata])
                if len(picks) > 1:
                    ax.plot([picks[-2][0], picks[-1][0]],[picks[-2][1], picks[-1][1]], "w-")
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            if event.button == 3:
                plt.close()

        fig.canvas.mpl_connect('button_press_event', onpick)

        plt.show()

        # check if the correct number of picks (at least three) has been made
        if len(picks)<3:
            raise Exception('At least three picks must be made!!!')

        # convert picks to polygon
        # define meshgrid of points
        x, y = np.meshgrid(np.arange(self.numX), np.arange(self.numY))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        p = Path(picks)
        grid = p.contains_points(points)
        self.polygon = grid.reshape(self.numY, self.numX)
        self.polygon = self.polygon.astype(float)

        # save out as an image
        imsave(name+'.jpg', self.polygon)

        # save out as npy
        np.save(name+'.npy', self.polygon)
