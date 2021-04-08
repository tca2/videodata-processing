# TODO: Probably eventually incorporate this into tracking_analysis.py, but for now it is convenient
# to have it separate for the sake of developing individual pieces without re-running everything.
import pandas as pd
from tqdm import tqdm


df = pd.read_csv('14_11_10_Green_20to21min.csv-match_indices.csv-ids.csv')


# Merge skeletons in the same frame with the same person_id
keypoints = [c.replace('_detected', '') for c in df if c.endswith('_detected')]
drop_ids = []
for _, frame_df in tqdm(df.groupby('frame_num'), 'Merging skeletons', total=df.frame_num.nunique()):
    id_counts = frame_df.person_id.value_counts()
    for person_id, _ in id_counts[id_counts > 1].items():
        skeleton_rows = frame_df[frame_df.person_id == person_id]
        merged_keypoints = {}
        for _, row in skeleton_rows.iterrows():
            for kp in keypoints:
                if row[kp + '_detected'] and row[kp + '_closest_dist'] < 15 and \
                        row[kp + '_second_closest_dist'] > 15:
                    assert kp + '_detected' not in merged_keypoints, \
                        'Tried to merge conflicting skeletons: frame=' + \
                        str(frame_df.frame_num.iloc[0]) + ', person ID=' + person_id
                    merged_keypoints[kp + '_closest_dist'] = row[kp + '_closest_dist']
                    merged_keypoints[kp + '_second_closest_dist'] = row[kp + '_second_closest_dist']
        if merged_keypoints:  # Replace first occurrence with unified skeleton, drop the rest
            df.at[skeleton_rows.index[0], merged_keypoints.keys()] = merged_keypoints.values()
            drop_ids.extend(skeleton_rows.index[1:])
orig_row_count = len(df)
df.drop(index=drop_ids, inplace=True)
print('Merged split skeletons and dropped', orig_row_count - len(df), 'rows')

# TODO: Interpolate missing skeletons
