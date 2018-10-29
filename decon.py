import numpy as np

# create a Ricker wavelet
def ricker(f, time, u):
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*((time - u)**2)) * np.exp(-(np.pi**2)*(f**2)*((time - u)**2))
    y /= np.sum(y)
    return y

# utility to find the next integer power of two of the input
def nextpow2(input):
    return int(2 ** np.ceil(np.log(input) / np.log(2.0)))

''' input and PSF are 1D list or numpy arrays.
    input is the data to be deconvolved and PSF is the function to deconvolve
    out of input.
    waterLevel is an optional argument to add a water level, in dB below peak,
    to the deconvolution.
    Deconvolution is carried out by frequency-domain division.
'''
def decon(input, PSF, waterLevel=None):
    # convert input and PSF to frequency domain
    lenInput = len(input)
    NFFT = nextpow2(np.max([lenInput, len(PSF)]))

    # convert input and PSF into frequency domain
    Finput = np.fft.fft(input, n=NFFT)
    FPSF = np.fft.fft(PSF, n=NFFT)

    # Fourier domain deconvolution
    Fout = Finput / FPSF

    # amend deconvolution based on water level, if one is provided
    if waterLevel is not None:
        spectrumPSF = np.abs(FPSF)
        waterLevelTrue = np.max(spectrumPSF) * (10.0 ** (-waterLevel / 20.0))
        Fout[FPSF==0] = waterLevelTrue
        indsBelowWaterLevel = spectrumPSF < waterLevelTrue
        FPSFMod = (waterLevelTrue * FPSF[indsBelowWaterLevel] / spectrumPSF[indsBelowWaterLevel])
        Fout[indsBelowWaterLevel] = Finput[indsBelowWaterLevel] / FPSFMod

    # convert deconvolved trace back to time domain
    output = np.real(np.fft.ifft(Fout))
    output = output[:lenInput]

    return output
