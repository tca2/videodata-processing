# To-do: 
# 1. account for target frames when there are not 1800 frames before and after (e.g., frame number 100) - should not exist, unless from a trimmed video

"""
This script takes as input a mp4 video file and a corresponding list of target frames 
and creates a single video of target video segments stitched together. Target video
segments are defined as videos with 60 seconds before and after the target frame resulting in 
a ~ 2-minute video clip. Corresponding time stamps are shown on the upper left side of each target 
video segment.

Dependencies: 
   -ffmpeg-python ('pip install ffmpeg-python') 
   -FFmpeg ('brew install ffmpeg' on Mac) and set to $PATH environment variable
       -can check if $PATH is set correctly by typing 'ffmpeg' into terminal
"""

import os
import ffmpeg
import glob


##### Part 1 #####
    # Take input video file, create a directory (segment_dir) to hold video segments,
    # and convert target frames list items into seconds to avoid separately slicing audio.

#video_file = input('Name of video file:\n')
video_file = None # NEED TO INPUT PATH TO VIDEO FILE TO BE TRIMMED
os.mkdir(video_file + '_target_segments')
segment_dir = video_file + '_target_segments'

in_file = ffmpeg.input(video_file)

#list_of_targetframes = input('Frames list:\n')
list_of_targetframes = None # NEED TO INPUT A LIST OF FRAMES TO USE AS TARGET FRAMES
timestamps = [int(x/30) for x in list_of_targetframes]

##### Part 2 #####
    # For each item in list_of_targetframes cut 1 minute before and after, use drawtext to save a 
    # timestamp in the upper left corner, and save video segments to segment_dir

for time in timestamps:
    
    starting_time = time - 60
    ending_time = time + 60
    (
        ffmpeg
        .input(video_file)
        .trim(
            start = starting_time,
            end = ending_time
        )
        .drawtext(
            text = str(starting_time) + '-' + str(ending_time),
            fontcolor = 'white',
            fontsize = 22,
            x = 0,
            y = 0
        )
        .setpts('PTS-STARTPTS')
        .output(segment_dir + '/' + str(starting_time) + '-' + str(ending_time) + \
            '_' + video_file)
        .run()
    )

##### Part 3 #####
    # Create a text file (filename.txt) of video segments names currently in segment_dir
    # then use it to concatenate them and save to current dir (should be one upper)

with open(video_file + '.txt', 'w') as f:
    for filename in glob.glob(segment_dir+'/*'):
        f.write('file '+ '\'' + filename + '\'' + '\n')

(
    ffmpeg
    .input(video_file+'.txt',f='concat', safe=0)
    .output('merged_video_segments_' + video_file)
    .run()
)