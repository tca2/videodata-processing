import pandas as pd
import glob
import os

#This script concatenates JSON files in a user defined directory into one large JSON file. It attempts to save the sourcefile (filename), framenumber, and keypoint information.
user_defined_dir = input("Please note that the concatenated output file will be saved to the directory one above the defined directory. Please define the directory by entering the full path to the directory that includes OpenPose JSON output files (example: C:\\Users\\yourname\\Documents\\json_raw\\): \n")
directory = glob.glob(user_defined_dir + "*")

for folder in directory:
    os.chdir(folder)
    listoflists = []
    for file in glob.glob(folder+"\*.json"):
        #sourcefile = '_'.join(file.split('_')[18:-2])
        sourcefile = file
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