#!/bin/bash
gnome-terminal - t "personstrack1" -x bash -c "cd /home/cv/Documents/wyw/PersonsTrack;/home/cv/anaconda3/envs/pt/bin/python run1_1_use_yolov3_with_data_fusion.py;exec bash"

gnome-terminal - t "personstrack2" -x bash -c "cd /home/cv/Documents/wyw/PersonsTrack;/home/cv/anaconda3/envs/pt/bin/python run2_1_use_openpose_with_data_fusion.py;exec bash"
gnome-terminal - t "personstrack3" -x bash -c "cd /home/cv/Documents/wyw/PersonsTrack;/home/cv/anaconda3/envs/pt/bin/python run2_2_use_openpose_with_data_fusion.py;exec bash"
gnome-terminal - t "personstrack4" -x bash -c "cd /home/cv/Documents/wyw/PersonsTrack;/home/cv/anaconda3/envs/pt/bin/python run2_3_use_openpose_with_data_fusion.py;exec bash"



