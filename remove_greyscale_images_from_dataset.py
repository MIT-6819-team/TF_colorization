#!/usr/bin/env python3
import glob, subprocess, itertools, os, cv2, sys
import numpy as np
from joblib import Parallel, delayed


# Winter Guerra <winterg@mit.edu> Nov. 27 2016

# Make sure that the user gave us enough arguments
assert len(sys.argv) == 3, "Not enough arguments. EX. ./remove_greyscale_images_from_dataset.py /path/to/trainingset/ .JPEG"
dataset_location = sys.argv[1]
file_extension = sys.argv[2]
num_jobs=3

print("Looking for files of type", file_extension, "from location", dataset_location)

def get_saturation(f):
	# Load image
	img = cv2.imread(f)
	# Convert BGR to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# Average the S layer
	S_avg = np.average(hsv[:,:,1])/255.0 # 255 since that is the bitdepth
	return S_avg

# Get list of files
filenames = glob.iglob(dataset_location + "*/*" + file_extension)

def worker_func(f):
	# get saturation values for all files
	saturation = get_saturation(f)
	
	# Annotate the filenames of all files with their rough saturation value.
	print((saturation, f))
	#os.rename(f, f[:-10])
	return

# iterate in parallel and remove said files
Parallel(n_jobs=num_jobs, verbose=5)(delayed(worker_func)(f) for f in filenames)
#print(list(itertools.islice(filenames, 5)))
