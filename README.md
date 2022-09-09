# videodata-processing

This repository contains files for processing [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) JSON output. Background for this work, including motivation and results, are written in detail in our paper ["Tracking Individuals in Classroom Videos via Post-processing OpenPose Data"](https://dl.acm.org/doi/10.1145/3506860.3506888), which was presented at the International Learning Analytics and Knowledge (LAK) Conference 2022. 

## Preparation

If video files should be concatenated before running OpenPose (e.g. one video is divided into many different video files), use `make_ffmpeg_concat.py`, which will create individual `.bat` files for the respective video files. The resulting `.bat` files will rely on the usage of [FFmpeg](https://ffmpeg.org) to create concatenated `.mov` files.

Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) processing commands for your project video file using the 25-keypoint body/foot keypoint configuration, and saving the respective output keypoint files in JSON format to a directory on your computer. 

Use `concat_JSON_files.py` to concatenate all JSON files into a formatted single `'FILENAME.csv'` file with a descriptive header row. `concat_JSON_files.py` asks the user for the directory that holds JSON output, and then saves the concatenated file to the directory above the user-defined directory. 

## Processing

Tracking processing is completed through `allprocesses_command.py`, and arguments are handled through Python argparse. More details about possible arguments can be found toward the end of the script. Currently the "--only_track" argument must be used, as OpenPose processing code is yet to be integrated. 

Processing will include calculating the closest distance (in pixels), closest distance owner (row), and second closest distance (in pixels), then subsequently tracking the skeletons and assigning IDs. In order to partially account for OpenPose's inconsistent detection of skeletons throughout frames.

## Referring to this work

If you would like to cite our work, please use the following citation (APA):

Hur, P., & Bosch, N. (2022, March). Tracking Individuals in Classroom Videos via Post-processing OpenPose Data. In LAK22: 12th International Learning Analytics and Knowledge Conference (pp. 465-471).
