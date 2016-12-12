#!/bin/bash
 
rclone copy './network/tests' 'Dropbox MIT:MIT/a_classes/6.819/final_project/tests' -v --update --size-only
