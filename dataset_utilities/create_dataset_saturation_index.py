#!/usr/bin/env python3
import glob2, itertools, cv2, sys, ujson, gzip
import numpy as np
from joblib import Parallel, delayed


# Winter Guerra <winterg@mit.edu> Nov. 27 2016

# Make sure that the user gave us enough arguments
assert len(sys.argv) == 3, "Not enough arguments. EX. ./<script>.py /path/to/trainingset/ .JPEG"
dataset_location = sys.argv[1]
file_extension = sys.argv[2]
num_jobs=4

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
filenames = glob2.iglob(dataset_location + "**/*" + file_extension)

# DEBUG
#filenames = itertools.islice(filenames, 5)

def worker_func(f):
	# get saturation values for all files
	saturation = get_saturation(f)
	
	# Get filename of file relative to the dataset root
	relative_filename = f[len(dataset_location):]

	return (relative_filename, saturation)

# iterate in parallel and remove said files
results = Parallel(n_jobs=num_jobs, verbose=5, backend='threading')(delayed(worker_func)(f) for f in filenames)

# Turn the results into a dictionary
file_dict = dict(results)

# Now, print the results to a binary json file
with gzip.open('./saturation_index.json.gz', 'wt') as f:

	ujson.dump(file_dict, f, double_precision=4)

#print(list(itertools.islice(filenames, 5)))
