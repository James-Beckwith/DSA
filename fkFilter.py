import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.path import Path

'''Class to define, store and apply FK filter to a 2D numpy arrayself.'''
class fkFilter:

    ''' utility to find the next integer power of two of the input '''
    def nextpow2(self, input):
        return int(2 ** np.ceil(np.log(input) / np.log(2.0)))

    def extents(self, f):
      delta = f[1] - f[0]
      return [f[0] - delta/2, f[-1] + delta/2]

    def __init__(self, inputData, sampleRateTime, sampleRateSpace, rampLength = 10, rampFunc = None, fkDomainFilter = None):
        ''' Initialization.
            inputData - 2D numpy array to apply FK filter toself.
                        shape of inputData = [# time sample, # of sapce samples].
            sampleRateTime - sample rate in time in seconds
            sampleRateSpace - sample rate in space in meters
            rampLength - length of linear ramp to apply at edge of FK domain picks
            rampFunc - pre-defined ramp function (1D numpy array) to be applied at
                       edge of FK domain picks
            fkDomainFilter - filter window defined in FK space
        '''

        # if a ramp function is input then store it, else build a linear ramp
        if rampFunc is not None:
            self.rampFunc = rampFunc
        else:
            self.rampFunc = np.arange(0, 1.0+1.0/(rampLength - 1), 1.0/(rampLength - 1))

        # store size of input data
        self.numberTime, self.numberSpace = np.shape(inputData)

        self.fkDomainFilter = fkDomainFilter
        self.inputData = inputData
        self.sampleRateTime = sampleRateTime
        self.sampleRateSpace = sampleRateSpace

        # Convert input data into FK domain
        # find lengths of frequency axes
        self.numberFrequency = self.nextpow2(self.numberTime)
        self.numberWavenumber = self.nextpow2(self.numberSpace)

        # first FK transform input data
        self.inputDataFK = np.fft.fft2(self.inputData, s=[self.numberFrequency, self.numberWavenumber])
        self.inputDataFK = self.inputDataFK

        # define frequency axes
        tempAddition = 1.0 / (self.numberFrequency / 2.0 + 1.0)
        self.freqTimePositive = 1.0 / (2.0 * self.sampleRateTime) * np.arange(0.0, 1.0 + tempAddition, tempAddition)
        tempAddition = 1.0 / (self.numberWavenumber)
        self.wavenumber = 1.0 / (2.0 * self.sampleRateSpace) * np.arange(-1.0 - tempAddition, 1.0 + tempAddition, tempAddition)


    ''' function to pick a window based on the FK domain response of the input data '''
    def pickWindow(self):

        # plot FK amplitude reponse for positive frequencies only
        fig, ax = plt.subplots()
        plt.imshow(np.abs(self.inputDataFK[(len(self.freqTimePositive) - 1):,:]), aspect='auto', interpolation='none',
            extent=self.extents(self.wavenumber) + self.extents(self.freqTimePositive), picker=True)

        picks=[]
        # define what to do when a pick is made, mouse button 1-> store picks,
        # mouse button 3->stop picking
        def onpick(event):
            if event.button==1:
                picks.append((event.xdata, event.ydata))
                if len(picks)>1:
                    ax.plot([picks[-2][0], picks[-1][0]],[picks[-2][1], picks[-1][1]],"w-")
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            if event.button==3:
                plt.close()

        fig.canvas.mpl_connect('button_press_event', onpick)

        plt.show()

        # convert picks to polygon
        # define meshgrid of points
        x, y = np.meshgrid(self.wavenumber, self.freqTimePositive,)
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T
        p = Path(picks)
        grid = p.contains_points(points)
        FKpolygon = grid.reshape(len(self.freqTimePositive), len(self.wavenumber))
        FKpolygon.astype(float)

        plt.figure()
        plt.imshow(FKpolygon, aspect='auto')
        plt.show()

        # apply ramp function to FKpolygon


    ''' apply pre-defined FK window function in FK domain and inverse FFT '''
    def applyFKFilter(self):

        # check if an FK domain window has been defined
        if self.fkDomainFilter is None:
            raise NameError('No FK domain window function defined')

    ''' create the impulse response of the FK domain filter in the time-space domain '''
    def impulseResponse(self):

        # check if an FK domain window has been defined
        if self.fkDomainFilter is None:
            raise NameError('No FK domain window function defined')
