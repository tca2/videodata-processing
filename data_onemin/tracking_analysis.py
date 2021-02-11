import glob
import re

import pandas as pd
from scipy.spatial import distance


# Calculate closest distance, closest distance owner, second closest distance, second closest
# distance owner
def calculate_distances(keypoint_dataframe, keypoint_num):
    for frame in keypoint_dataframe.groupby(['frame_num']):
        for index, row in frame[1].iterrows():  # for each person in current frame
            closest_dist = None
            second_closest_dist = None
            morethan0 = False
            index_closest = None
            if list(row[1:3]) != [0, 0]:  # If x and y are 0, OpenPose didn't detect the keypoint
                for indexn, rown in keypoint_dataframe[keypoint_dataframe.frame_num == (int(frame[0]+1))].iterrows():
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
                print(f'(Closest distance, index) = ({closest_dist},{index_closest})')
                print('Second closest distance is', second_closest_dist)
            else:
                print(f'{keypoint_num[3:]} was not detected by OpenPose')


fileslist = glob.glob('*.csv')
print('Running script on these .csv files in current directory:\n', fileslist, '\n')
for file in fileslist:
    df = pd.read_csv(file)
    dict_of_keypoint_dfs = {f'df_{col[:-2]}': df[['frame_num', col, col.replace('x', 'y')]]
                            for col in df.columns if re.search(r'^keypoint\d*_x', col)}
    for key, value in dict_of_keypoint_dfs.items():
        print(key)
        calculate_distances(value, key)
        break
