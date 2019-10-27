import numpy as np
from PIL import Image
from imagePicking import imagePicking
import argparse
import sys

parser = argparse.ArgumentParser(description='Function used to pick masks on images interactively. \
                                             An input image filepath can be given with "--img" otherwise\
                                             one will be asked for')
parser.add_argument('--img', dest='imagePath', default=None,
                    help='Filepath for image to be used to pick mask')
args = parser.parse_args()

if args.imagePath is None:
    print('Give full path to input image (remember to escape the necessary characters!')
    imagePath = input()
else:
    imagePath = args.imagePath

# read and convert to greyscale
im = np.asarray(Image.open(imagePath).convert('L'))

print('Please give a name for the mask, it will be appended to the input file name')
name = input()
# Instantiate imagepicking object
impick = imagePicking(im)
# pick windows
impick.pickWindow(imagePath + '_' + name)

while 1>0:
    print('Enter a name for another window or type "N" to exiting picking')
    finished = input()
    if finished == 'N':
        break
    impick.pickWindow(imagePath + '_' + finished)


