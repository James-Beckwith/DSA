import numpy as np
import matplotlib.pyplot as plt

'''Class to define, store and apply FK filter to a 2D numpy arrayself.'''
class fkFilter:

    ''' utility to find the next integer power of two of the input '''
    def nextpow2(self, input):
        return int(2 ** np.ceil(np.log(input) / np.log(2.0)))

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
        plt.figure
        plt.imshow(np.abs(self.inputDataFK[(len(self.freqTimePositive) - 1):,:]), aspect='auto', interpolation='none',
           extent=extents(self.wavenumber) + extents(self.freqTimePositive))

        plt.show()

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
