# videodata-processing

This repository contains files for processing [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) output.

## Preparation

If video files should be concatenated before running OpenPose (e.g. one video is divided into many different video files), use `make_ffmpeg_concat.py`, which will create individual `.bat` files for the respective video files. The resulting `.bat` files will rely on the usage of [FFmpeg](https://ffmpeg.org) to create concatenated `.mov` files.

Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) processing commands for your project video file using the 25-keypoint body/foot keypoint configuration, and saving the respective output keypoint files in JSON format to a directory on your computer. 

Use `Concatenater.ipynb` (**TODO - upload replacement .py file**) to concatenate all JSON files into a formatted single `'FILENAME.csv'` file with a descriptive header row. 

## Processing

Calculate relevant values for tracking using `data_onemin/tracking_analysis.py` on the concatenated `.csv` file. This will calculate the closest distance (in pixels), closest distance owner (row), and second closest distance (in pixels) and append them to the data rows, and save as a new file, `'FILENAME-match_indices.csv'`.

With `FILENAME-match_indices.csv` as input, skeletons are able to be tracked and assigned IDs using `assign_ids.py`. It checks backwards in frames (up to 5 frames) to find the skeleton with the closest match (within 15 pixels distance for compared keypoints), and if there is no close match, assigns a new ID. The result is `'FILENAME-match_indices.csv-ids.csv'`

In order to account for OpenPose's inconsistent detection of skeletons throughout frames, missing skeletons are interpolated in `postprocess_skeletons.py` using `'FILENAME-match_indices.csv-ids.csv'` as input, and outputs a named `.csv` file (must manually name).


**As a summary, the processing files are run in the following order:**
1. `Concatenater.ipynb`
2. `data_onemin/tracking_analysis.py`
3. `assign_ids.py`
4. `postprocess_skeletons.py`

Future development will work to merge these components together.
