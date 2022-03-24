# Prerequisite: 
#   Requires OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose). Download windows portable version and save to a easily accessile location, and note the full file path to the 'OpenPoseDemo.exe' file.

import glob 
import os
import os.path
import subprocess
import glob
import re
from collections import deque
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm
from collections import defaultdict

# PROCESS 1: ASK FOR NAME OF DIRECTORY TO RECURSIVELY SEARCH WITHIN. SHOW FIRST 5 VIDEOS (AND TELL HOW MANY VIDEOS WERE FOUND), ALLOW USER TO INPUT BETWEEN 1-5, OR TYPE ALL FOR ALL VIDEOS. RUN OPENPOSE ON VIDEOS WITHIN USER-DEFINED DIRECTORY.
#           Implement option to run on one particular video

def run_OpenPose(dir_name, openpose_dir):
    print('\n', 'Running OpenPose process...')

    src = input('Please enter the directory path where the files are located: ')
    if not os.path.isdir(src):
        print('Invalid directory path, please check your directory path and re-enter.')
        exit(1)
    path = os.path.join(src)
    for f in glob.glob(path):
        new_filename = '203 Reference Letter {}'.format(name)
        os.rename(f, new_filename)

    videotypes = ('*.mov', '*.mp4', '*wmv', '*avi', '*flv', '*mkv') # accepted video types
    files_grabbed = []
    for files in videotypes:
        files_grabbed.extend(glob.glob(dir_name,files))

    fileslist = glob.glob('*.mov')
    print('Running OpenPose on the following .MOV files in current directory (make sure that OpenPose models folder is in the current directory)):\n', fileslist, '\n')

    OpenPoseEXEPath = input("Pleae enter the full path to the OpenPoseDemo.exe. It should look something like 'DRIVE:\\USER\\DOCUMENTS\\OPENPOSEFOLDER\\OPENPOSE\\BIN\\OpenPoseDemo.exe'.")
    #'C:\\Users\\khur4\\Documents\\openpose-1.6.0-binaries-win64-gpu-flir-3d_recommended\\openpose\\bin\\OpenPoseDemo.exe' 

    #file_txt = open('file_ids.txt','w+')

    for file in fileslist:
        os.mkdir('file_'+str(fileslist.index(file))+'_videoframes')
        os.mkdir('file_'+str(fileslist.index(file))+'_JSON')
        #file_txt.write('file_'+str(fileslist.index(file))+' = '+str(file)+'\n')
        #print('Running: \n',OpenPoseEXEPath, '--video', file, '--write_images', 'file_'+str(fileslist.index(file))+'_videoframes', '--write_json', 'file_'+str(fileslist.index(file))+'_JSON','\n')
        subprocess.run([OpenPoseEXEPath, '--video', file, '--write_images', 'file_'+str(fileslist.index(file))+'_videoframes', '--write_images_format','jpg','--write_json', 'file_'+str(fileslist.index(file))+'_JSON', '\n'])
        #subprocess.run([OpenPoseEXEPath, '--video', file, '--write_images', 'file_'+str(fileslist.index(file))+'_videoframes', '\n'])

    #file_txt.close()
    
    print('\n', 'Running OpenPose process complete.')
    
    #return

# PROCESS 1A: Take the OpenPose JSON output and concatenate it into singular CSV file per video 

def concat_JSON(argu):
    directory = glob.glob('C:\\Users\\khur4\\Documents\\one_minute_sampler\\Teacher\\_tracking_analysis\\json_raw\\*')
    for folder in directory:
        os.chdir(folder)
        listoflists = []
        for file in glob.glob(folder+"\*.json"):
            sourcefile = '_'.join(file.split('_')[18:-2])
            framenum = file.split('_')[-2]
            data = pd.read_json(file)
            for person in data['people']:
                personlist = []
                personlist.extend([sourcefile,framenum,-1])
                for i in person['pose_keypoints_2d']:
                    personlist.extend([i])
                listoflists.append(personlist)
        df = pd.DataFrame.from_records(listoflists)
        #Rename Columns
        col_names = ['source_file', 'frame_num', 'person_id']
        for i in range(1, 26):
            col_names.extend(['keypoint' + str(i) + '_x', 'keypoint' + str(i) + '_y', 'keypoint' + str(i) + '_conf'])
        df.columns = col_names

        #Create CSV file to directory one above current directory to prevent the headache of moving files later
        df.to_csv('../'+sourcefile+'.csv')

# PROCESS 2: Calculate closest distance, closest distance owner, second closest distance, then track

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

'''
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
'''
'''
# TODO: Probably eventually incorporate this into tracking_analysis.py, but for now it is convenient
# to have it separate for the sake of developing individual pieces without re-running everything.
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
import glob

#directory = glob.glob('C:\\Users\\khur4\\Documents\\one_minute_sampler\\Teacher\\_tracking_analysis\\*match_indices.csv')
directory = glob.glob('*match_indices.csv')


for file in directory:
    df = pd.read_csv(file)
    keypoints = [c.replace('_detected', '') for c in df.columns if c.endswith('_detected')]

    # Precalculate sets of detected keypoints to improve speed
    detected_kps = {i: set([k for k in keypoints if r[k + '_detected']])
                    for i, r in tqdm(df.iterrows(), 'Counting detected keypoints', total=len(df))}
    # Go through frames in reverse order to follow links to earlier frames. Then, if we reach a frame
    # that has no frames linked to it, assign a new person ID. Alternatively, if we reach a frame that
    # has multiple frames linked to it, resolve the ambiguity via voting for the most keypoints matched.
    proposed_ids = defaultdict(set)
    df.insert(2, 'person_id', '')
    df.insert(3, 'new_id', 0)
    for row_i, row in df[::-1].iterrows():
        # Select keypoints that were good enough matches to participate in the overall match decision
        ambiguous_kps = [k for k in keypoints if row[k + '_second_closest_dist'] < 15]
        voting_kps = [k for k in keypoints if row[k + '_closest_dist'] < 15 and k not in ambiguous_kps]

        # First try to assign a final ID to the current row based on its matches
        if row_i in proposed_ids.keys():
            if len(proposed_ids[row_i]) == 1:  # Clear success
                df.at[row_i, 'person_id'] = proposed_ids[row_i].pop()[0]
            else:  # Resolve ambiguous match by matching to best match (most votes)
                ranked_ids = sorted(proposed_ids[row_i], key=lambda x: x[1], reverse=True)
                if ranked_ids[0] == ranked_ids[1]:
                    print('Extra-ambiguous match for row', row_i, ' -- matches:', proposed_ids[row_i])
                df.at[row_i, 'person_id'] = ranked_ids[0][0]
        else:  # No matches for this row; start a new ID
            df.at[row_i, ['person_id', 'new_id']] = ['idx' + str(row_i), 1]

        # Count votes for matches to other rows
        tally = row[[k + '_closest_index' for k in voting_kps]].value_counts()
        # Go through each match and count it as a match unless it conflicts with one of the matches with
        # a better match.
        used_kps = set()
        for match_i, num_votes in tally.iteritems():  # Descending (most votes first)
            # Keypoints in common among this person and the potential match
            common_kps = detected_kps[row_i].intersection(detected_kps[match_i])
            if common_kps.intersection(used_kps):
                continue  # Skip because a better match already used some of these keypoints
            if all(k in ambiguous_kps or k in voting_kps and row[k + '_closest_index'] == match_i
                for k in common_kps):
                # Set as a match if all the keypoints in common between the two were indeed matches
                if used_kps:
                    print('Found a split skeleton:', row_i)
                used_kps.update(common_kps)
                proposed_ids[match_i].add((df.at[row_i, 'person_id'], num_votes))

    print('Saving')
    df.to_csv(file+'-ids.csv', index=False)

'''
def run_allprocesses(dir_name):
    run_OpenPose(dir_name)

