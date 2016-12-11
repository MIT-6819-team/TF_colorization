#!/bin/bash

echo "Syncing network/tests to Winter's shared Dropbox"

rclone sync -v network/tests "Dropbox MIT:MIT/a_classes/6.819/final_project"
