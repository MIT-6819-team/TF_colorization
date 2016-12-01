#!/usr/bin/env python3
import glob, subprocess, itertools, os, cv2
import numpy as np
from joblib import Parallel, delayed


# Winter Guerra <winterg@mit.edu> Nov. 27 2016

# Minimum saturation
min_saturation = 0.05
dataset_location = "/root/persistant_data/datasets/places_2/train_256/"
num_jobs=3

def get_saturation(f):
        #command = "convert {} -colorspace HSL -channel g -separate +channel -format '%[fx:mean]' info:".format(f)
        #op = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
        #return float(op.stdout)

	# Load image
	img = cv2.imread(f)
	# Convert BGR to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# Average the S layer
	S_avg = np.average(hsv[:,:,1])
	return S_avg

# Get list of files
filenames = glob.iglob(dataset_location + "**/*.jpg.greyscale", recursive=True)

def worker_func(f):
	# get saturation values for all files
	saturation = get_saturation(f)
	
	# Annotate the filenames of all files with their rough saturation value.
	print((saturation, f))
	#os.rename(f, f[:-10])
	return

# iterate in parallel and remove said files
Parallel(n_jobs=num_jobs, verbose=5)(delayed(worker_func)(f) for f in filenames)
#print(list(itertools.islice(low_saturation_results, 5)))
