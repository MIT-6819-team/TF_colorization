#!/usr/bin/env python3
import glob, subprocess, itertools, os
from joblib import Parallel, delayed


# Winter Guerra <winterg@mit.edu> Nov. 27 2016

# Minimum saturation
min_saturation = 0.01
dataset_location = "/media/data/datasets/places_2/train_256_rgb_only/"
num_jobs=3

def get_saturation(f):
        command = "convert {} -colorspace HSL -channel g -separate +channel -format '%[fx:mean]' info:".format(f)
        op = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
        return float(op.stdout)


# Get list of files
filenames = glob.iglob(dataset_location + "**/*.jpg", recursive=True)

def worker_func(f):
	# get saturation values for all files
	saturation = get_saturation(f)

	# filter results to only include files below our threshold
	if saturation <= min_saturation:
		os.rename(f, f+".greyscale")
	return

# iterate in parallel and remove said files
Parallel(n_jobs=num_jobs, verbose=5)(delayed(worker_func)(f) for f in filenames)
#print(list(itertools.islice(low_saturation_results, 5)))
