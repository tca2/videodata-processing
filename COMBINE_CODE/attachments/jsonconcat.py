import pandas as pd
import glob

#Concatenate All Json Files in Directory
listoflists = []
for file in glob.glob("*.json"):
    sourcefile = '_'.join(file.split('_')[8:-2])
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