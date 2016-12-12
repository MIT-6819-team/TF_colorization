#!/bin/bash

echo "Syncing network/tests to Winter's shared Dropbox every 10 minutes."

#watch -n 600 -- 
rclone sync network/tests 'Dropbox\ MIT:MIT/a_classes/6.819/final_project/' -v --transfers 20 
