import glob
import re
from collections import deque

import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm


# Calculate closest distance, closest distance owner, second closest distance
def calculate_distances(keypoint_df, keypoint_num):
    matches = pd.DataFrame(index=keypoint_df.index)
    matches['frame_num'] = keypoint_df.frame_num
    # Count low-confidence detections as non-detections because they can be misleading
    matches[keypoint_num + '_detected'] = \
        (keypoint_df[keypoint_num + '_conf'] > .3).astype(int)
    prev_frames = deque(maxlen=5)  # Number of frames into the past to search for matches
    # Iterate over only successful detection rows to prevent matches to low-confidence skeletons
    frame_iter = iter(keypoint_df[matches[keypoint_num + '_detected'] == 1].groupby(['frame_num']))
    prev_frames.append(next(frame_iter)[1])  # Skip to starting on second frame
    for _, frame in tqdm(frame_iter, total=keypoint_df.frame_num.nunique() - 1,
                         desc='Matching ' + str(keypoint_num)):
        for row_i, row in frame.iterrows():  # for each person in current frame
            closest_dist = None
            second_closest_dist = None
            index_closest = None
            for lag in range(-1, -len(prev_frames) - 1, -1):  # Look backward in time
                for indexn, rown in prev_frames[lag].iterrows():
                    dist = distance.euclidean([row[1:3]], [rown[1:3]])
                    if closest_dist is None or dist < closest_dist:
                        second_closest_dist = closest_dist
                        closest_dist = dist
                        index_closest = indexn
                if closest_dist < 15:
                    break  # Close enough; stop looking further back
            matches.at[row_i, [keypoint_num + '_closest_index', keypoint_num + '_closest_dist',
                               keypoint_num + '_second_closest_dist']] = \
                [index_closest, closest_dist, second_closest_dist]
        prev_frames.append(frame)
    return matches


fileslist = glob.glob('*.csv')
print('Running script on these .csv files in current directory:\n', fileslist, '\n')
for file in fileslist:
    df = pd.read_csv(file)
    keypoints = [c.replace('_x', '') for c in df.columns if re.search(r'^keypoint\d*_x', c)]
    keypoint_dfs = {c: df[['frame_num', c + '_x', c + '_y', c + '_conf']] for c in keypoints}
    result_df = pd.DataFrame(index=df.index)
    for key, value in keypoint_dfs.items():
        dist_df = calculate_distances(value, key)
        # TODO: Change all the input/output filenames and parameters to argparse
        # TODO: Change the arbitrary 15px distance threshold to something auto-detected
        result_df[dist_df.columns] = dist_df
        result_df.to_csv(file + '-match_indices.csv', index_label='orig_index')
