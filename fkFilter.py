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
            self.rampFunc = np.arange(1, 0.0-1.0/(rampLength - 1), -1.0/(rampLength - 1))

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
        tempAddition = 1.0 / (self.numberFrequency / 2.0)
        self.freqTimePositive = 1.0 / (2.0 * self.sampleRateTime) * np.arange(0.0, 1.0 + tempAddition, tempAddition)
        tempAddition = 1.0 / ((self.numberWavenumber - 1.0)/ 2.0)
        self.wavenumber = 1.0 / (2.0 * self.sampleRateSpace) * np.arange(-1.0, 1.0 + tempAddition, tempAddition)


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
                picks.append([event.xdata, event.ydata])
                if len(picks)>1:
                    ax.plot([picks[-2][0], picks[-1][0]],[picks[-2][1], picks[-1][1]],"w-")
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            if event.button==3:
                plt.close()

        fig.canvas.mpl_connect('button_press_event', onpick)

        plt.show()

        # convert pick data to sample locations
        picksSampX = [(np.abs(picks[i][0] - self.wavenumber)).argmin() for i in range(len(picks))]
        picksSampY = [(np.abs(picks[i][1] - self.freqTimePositive)).argmin() for i in range(len(picks))]
        # combine data back together
        picksSamp = np.vstack((picksSampX, picksSampY)).T.tolist()

        # convert picks to polygon
        # define meshgrid of points
        x, y = np.meshgrid(np.arange(self.numberWavenumber), np.arange(len(self.freqTimePositive)))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T

        p = Path(picksSamp)
        grid = p.contains_points(points)
        self.FKpolygon = grid.reshape(len(self.freqTimePositive), len(self.wavenumber))
        self.FKpolygon = self.FKpolygon.astype(float)

        # apply ramp function if necessary
        if (self.rampFunc is not None):
            if len(self.rampFunc) > 1:
                self.applyRamp(picksSamp)
                self.FKpolygon += self.windowRamp
                self.FKpolygon[self.FKpolygon>1] = 1

        plt.figure()
        plt.imshow(self.FKpolygon, aspect='auto')
        plt.show()


    ''' apply ramp function to picked points '''
    def applyRamp(self, picks):

        # interpolate points onto grid samplingtion
        # 'ramp' is ramp length
        # interpolate between points (1D linear interpolation in 2D space)
        marks=0
        picks.append(picks[0])
        rampLength = len(self.rampFunc)
        line1=[]
        #tmp=[picks;picks(1,:)];
        for ii in range(len(picks) - 1):

            # fit straight line between successive points
            dx=picks[ii+1][0] - picks[ii][0]
            dy=picks[ii+1][1] - picks[ii][1]
            # try:
            #     m=dy/dx
            #     mInfFlag=0
            #     c=picks[ii][1] - m*picks[ii][0]
            # except:
            #     mInfFlag = 1

            if (dx!=0) or (dy!=0):
                if abs(dx)>=abs(dy):
                    m=dy/dx
                    c=picks[ii][1] - m*picks[ii][0]
                    tmp = np.arange(picks[ii][0], picks[ii+1][0] + dx/abs(dx), dx/abs(dx))
                    #line1.append([tmp.tolist(), (m*tmp+c).tolist()])
                    line1 += np.vstack([tmp.tolist(), (m*tmp+c).tolist()]).T.tolist()
                    marks=marks+abs(dx)
                else:
                    #line1(marks:marks+abs(dy),2)=tmp(ii,2):dy/abs(dy):tmp(ii+1,2);
                    tmp = np.arange(picks[ii][1], picks[ii+1][1] + dy/abs(dy), dy/abs(dy))
                    if dx==0:
                        #line1(marks:marks+abs(dy),1)=picks[ii][0]*len(tmp);
                        #line1.append([[picks[ii][0]]*len(tmp), tmp.tolist()])
                        line1 += np.vstack([[picks[ii][0]]*len(tmp), tmp.tolist()]).T.tolist()
                    else:
                        m=dy/dx
                        c=picks[ii][1] - m*picks[ii][0]
                        #line1(marks:marks+abs(dy),1)=(line1(marks:marks+abs(dy),2)-c)/m;
                        #line1.append([((tmp - c)/m).tolist(), tmp.tolist()])
                        line1 += np.vstack([((tmp - c)/m).tolist(), tmp.tolist()]).T.tolist()
                    marks=marks+abs(dy)

        # find lowest and highest poits of window in x and y
        line1AsArray = np.asarray(line1)
        minx=np.min(line1AsArray[:,0])
        maxx=np.max(line1AsArray[:,0])
        miny=np.min(line1AsArray[:,1])
        maxy=np.max(line1AsArray[:,1])

        self.windowRamp = np.zeros([len(self.freqTimePositive), self.numberWavenumber])

        # speed up window generation by only investigating points with a ramps
        # distance of the picked window boundary
        for ii in range(len(self.freqTimePositive)):
            if (ii + rampLength > miny) and (ii - rampLength < maxy):
                for jj in range(self.numberWavenumber):

                    if (jj + rampLength > minx) and (jj - rampLength < maxx):
                        if self.FKpolygon[ii,jj]!=1:
                            # define distance to nearest window edge
                            tmp=((line1AsArray[:,0] - jj)**2 + (line1AsArray[:,1] - ii)**2)**0.5
                            dist1=np.round(np.min(tmp))
                            if dist1==0:
                                self.windowRamp[ii,jj]=1
                            elif dist1 <= rampLength:
                                self.windowRamp[ii,jj]=self.rampFunc[int(dist1) - 1]

    ''' apply pre-defined FK window function in FK domain and inverse FFT '''
    def applyFKFilter(self):
        print('poop')

    ''' create the impulse response of the FK domain filter in the time-space domain '''
    def impulseResponse(self):

        # check if an FK domain window has been defined
        if self.fkDomainFilter is None:
            raise NameError('No FK domain window function defined')
