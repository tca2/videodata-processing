import glob
import re
from collections import deque

import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm


# Calculate closest distance, closest distance owner, second closest distance
def calculate_distances(keypoint_dataframe, keypoint_num):
    matches = pd.DataFrame(index=keypoint_dataframe.index)
    matches['frame_num'] = keypoint_dataframe.frame_num
    prev_frames = deque(maxlen=5)
    frame_iter = iter(keypoint_dataframe.groupby(['frame_num']))
    prev_frames.append(next(frame_iter)[1])  # Skip to starting on second frame
    for _, frame in tqdm(frame_iter, total=keypoint_dataframe.frame_num.nunique() - 1,
                         desc='Matching ' + str(keypoint_num)):
        for row_i, row in frame.iterrows():  # for each person in current frame
            closest_dist = None
            second_closest_dist = None
            morethan0 = False
            index_closest = None
            if list(row[1:3]) != [0, 0]:  # If x and y are 0, OpenPose didn't detect the keypoint
                for lag in range(-1, -len(prev_frames) - 1, -1):  # Look backward in time
                    for indexn, rown in prev_frames[lag].iterrows():
                        if not morethan0:
                            closest_dist = distance.euclidean([row[1:3]], [rown[1:3]])
                            index_closest = indexn
                            if closest_dist > 0:
                                morethan0 = True
                        if morethan0:
                            dist = distance.euclidean([row[1:3]], [rown[1:3]])
                            if dist < closest_dist:
                                second_closest_dist = closest_dist
                                closest_dist = dist
                                index_closest = indexn
                    if closest_dist < 10:
                        break  # Close enough; stop looking further back
            matches.at[row_i, ['closest_index', 'closest_dist', 'second_closest_dist']] = \
                [index_closest, closest_dist, second_closest_dist]
        prev_frames.append(frame)
    return matches


fileslist = glob.glob('*.csv')
print('Running script on these .csv files in current directory:\n', fileslist, '\n')
for file in fileslist:
    df = pd.read_csv(file)
    dict_of_keypoint_dfs = {f'df_{col[:-2]}': df[['frame_num', col, col.replace('x', 'y')]]
                            for col in df.columns if re.search(r'^keypoint\d*_x', col)}
    # TODO: We should probably filter the keypoints and only try to match on ones with few 0s
    # (OpenPose detection failures) to save a bit of computational power.
    for key, value in dict_of_keypoint_dfs.items():
        print(key)
        dist_df = calculate_distances(value, key)
        dist_df.to_csv(file + '-' + key + '.csv', index_label='orig_index')
        break
