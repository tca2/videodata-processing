# TODO: Probably eventually incorporate this into tracking_analysis.py, but for now it is convenient
# to have it separate for the sake of developing individual pieces without re-running everything.
import pandas as pd
import numpy as np
from tqdm import tqdm


# Load data
df = pd.read_csv('14_11_10_Green_20to21min.csv-match_indices.csv-ids.csv')
orig_df = pd.read_csv('14_11_10_Green_20to21min.csv')
assert(len(df) == len(orig_df))
keypoints = [c.replace('_detected', '') for c in df if c.endswith('_detected')]
coord_columns = [kp + '_x' for kp in keypoints] + [kp + '_y' for kp in keypoints]
df[coord_columns] = orig_df[coord_columns]


# Merge skeletons in the same frame with the same person_id
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
                    merged_keypoints[kp + '_x'] = row[kp + '_x']
                    merged_keypoints[kp + '_y'] = row[kp + '_y']
        if merged_keypoints:  # Replace first occurrence with unified skeleton, drop the rest
            df.at[skeleton_rows.index[0], merged_keypoints.keys()] = merged_keypoints.values()
            drop_ids.extend(skeleton_rows.index[1:])
orig_row_count = len(df)
df.drop(index=drop_ids, inplace=True)
print('Merged split skeletons and dropped', orig_row_count - len(df), 'rows')


# Interpolate missing skeletons
interpolated_dfs = []
for pid, pid_df in tqdm(df.groupby('person_id'), 'Interpolating', total=df.person_id.nunique()):
    # Use Pandas built-in interpolate by constructing a DataFrame for this person that includes
    # NaNs where appropriate for the rows to be interpolated
    if pid_df.frame_num.max() - pid_df.frame_num.min() == len(pid_df) - 1:
        # Nothing to interpolate
        interpolated_dfs.append(pid_df)
    else:
        new_df = pd.DataFrame(index=np.arange(pid_df.frame_num.min(), pid_df.frame_num.max() + 1))
        for col in pid_df.columns:
            new_df.insert(len(new_df.columns), col, pd.Series(dtype=pid_df[col].dtype))
        new_df.loc[pid_df.frame_num] = pid_df.values
        # Interpolate only the position columns we care about
        new_df[coord_columns] = new_df[coord_columns].interpolate()
        assert not new_df[coord_columns].isna().any().any(), 'Interpolating skeletons failed'
        interpolated_dfs.append(new_df)

# Merge interpolated values to form combined DataFrame
print('Saving')
interpolated_df = pd.concat(interpolated_dfs)
interpolated_df.to_csv('14_11_10_Green_20to21min.csv-match_indices.csv-ids.csv-postprocessed.csv',
                       index=False)
