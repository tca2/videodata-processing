import os
import argparse
import re
from collections import deque, defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm


# Calculate closest distance, closest distance owner, second closest distance
def calculate_distances(keypoint_df, keypoint_num, lookback):
    matches = pd.DataFrame(index=keypoint_df.index)
    matches['frame_num'] = keypoint_df.frame_num
    # Count low-confidence detections as non-detections because they can be misleading
    matches[keypoint_num + '_detected'] = \
        (keypoint_df[keypoint_num + '_conf'] > .3) * 1
    prev_frames = deque(maxlen=lookback)  # Number of frames into the past to search for matches
    # Iterate over only successful detection rows to prevent matches to low-confidence skeletons
    detected_df = keypoint_df[matches[keypoint_num + '_detected'] == 1]
    frame_iter = iter(detected_df.groupby(['frame_num']))
    try:
        prev_frames.append(next(frame_iter)[1])  # Skip to starting on second frame
    except StopIteration:  # No usable data for this keypoint
        pass
    row_indices = []
    closest_indices = []
    closest_dists = []
    second_closest_dists = []
    for _, frame in tqdm(frame_iter, total=detected_df.frame_num.nunique() - 1,
                         desc='Matching ' + str(keypoint_num)):
        for row_i, row in frame.iterrows():  # for each person in current frame
            closest_dist = None
            second_closest_dist = 10000000
            index_closest = None
            for lag in range(-1, -len(prev_frames) - 1, -1):  # Look backward in time
                x_diff = prev_frames[lag][keypoint_num + '_x'] - row[keypoint_num + '_x']
                y_diff = prev_frames[lag][keypoint_num + '_y'] - row[keypoint_num + '_y']
                # Calculate squared distances; no need to take square root
                two_closest = (x_diff * x_diff + y_diff * y_diff).nsmallest(2)
                if closest_dist is None or two_closest.iloc[0] < closest_dist:
                    closest_dist = two_closest.iloc[0]
                    index_closest = two_closest.index[0]
                    if len(two_closest) > 1:
                        second_closest_dist = two_closest.iloc[1]
                # Check versus number of pixels squared, since that is faster than square root
                if closest_dist < 15 * 15:  # TODO: Autodetect this or something
                    break  # Close enough; stop looking further back
            row_indices.append(row_i)
            closest_indices.append(index_closest)
            closest_dists.append(closest_dist)
            second_closest_dists.append(second_closest_dist)
        prev_frames.append(frame)
    matches.loc[row_indices, keypoint_num + '_closest_index'] = closest_indices
    matches.loc[row_indices, keypoint_num + '_closest_dist'] = closest_dists
    matches.loc[row_indices, keypoint_num + '_second_closest_dist'] = second_closest_dists
    return matches.drop(columns='frame_num')


# Based on a set of closest/second-closest distances, try to assign person IDs
def assign_person_ids(matches_df):
    df = matches_df.copy()
    # Precalculate sets of detected keypoints to improve speed
    keypoints = [c.replace('_closest_index', '') for c in matches_df.columns
                 if c.endswith('_closest_index')]
    detected_kps = {i: set([k for k in keypoints if r[k + '_detected']])
                    for i, r in tqdm(df.iterrows(), 'Counting detected keypoints', total=len(df))}
    # Go through frames in reverse order to follow links to earlier frames. Then, if we reach a
    # frame that has no frames linked to it, assign a new person ID. Alternatively, if we reach a
    # frame that has multiple frames linked to it, resolve the ambiguity via voting for the most
    # keypoints matched.
    proposed_ids = defaultdict(set)
    df.insert(1, 'person_id', '')
    for row_i, row in df[::-1].iterrows():
        # Select keypoints that were good enough matches to participate in the overall match vote
        ambig_kps = [k for k in keypoints if row[k + '_second_closest_dist'] < 15]
        voting_kps = [k for k in keypoints if row[k + '_closest_dist'] < 15 and k not in ambig_kps]

        # First try to assign a final ID to the current row based on its matches
        if row_i in proposed_ids.keys():
            if len(proposed_ids[row_i]) == 1:  # Clear success
                df.at[row_i, 'person_id'] = proposed_ids[row_i].pop()[0]
            else:  # Resolve ambiguous match by matching to best match (most votes)
                ranked_ids = sorted(proposed_ids[row_i], key=lambda x: x[1], reverse=True)
                if ranked_ids[0] == ranked_ids[1]:
                    print('Extra-ambiguous match for row', row_i, '- matches:', proposed_ids[row_i])
                df.at[row_i, 'person_id'] = ranked_ids[0][0]
        else:  # No matches for this row; start a new ID
            df.at[row_i, 'person_id'] = 'idx' + str(row_i)

        # Count votes for matches to other rows
        tally = row[[k + '_closest_index' for k in voting_kps]].value_counts()
        # Go through each match and count it as a match unless it conflicts with one of the matches
        # with a better match.
        used_kps = set()
        for match_i, num_votes in tally.iteritems():  # Descending (most votes first)
            # Keypoints in common among this person and the potential match
            common_kps = detected_kps[row_i].intersection(detected_kps[match_i])
            if common_kps.intersection(used_kps):
                continue  # Skip because a better match already used some of these keypoints
            if all(k in ambig_kps or k in voting_kps and row[k + '_closest_index'] == match_i
                   for k in common_kps):
                # Set as a match if all the keypoints in common between the two were indeed matches
                if used_kps:
                    print('Found a split skeleton:', row_i)
                used_kps.update(common_kps)
                proposed_ids[match_i].add((df.at[row_i, 'person_id'], num_votes))
    return df


def postprocess_ids(ids_df, orig_df, out_fname):
    df = ids_df.copy()
    keypoints = [c.replace('_closest_index', '') for c in df.columns
                 if c.endswith('_closest_index')]
    coord_columns = [kp + '_x' for kp in keypoints] + [kp + '_y' for kp in keypoints]
    df[coord_columns] = orig_df[coord_columns]

    # Merge skeletons in the same frame with the same person_id
    drop_ids = []
    for _, frame_df in tqdm(df.groupby('frame_num'), 'Merging skeletons',
                            total=df.frame_num.nunique()):
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
                        merged_keypoints[kp + '_second_closest_dist'] = \
                            row[kp + '_second_closest_dist']
                        merged_keypoints[kp + '_x'] = row[kp + '_x']
                        merged_keypoints[kp + '_y'] = row[kp + '_y']
            if merged_keypoints:  # Replace first occurrence with unified skeleton, drop the rest
                df.at[skeleton_rows.index[0], merged_keypoints.keys()] = merged_keypoints.values()
                drop_ids.extend(skeleton_rows.index[1:])
    orig_row_count = len(df)
    df.drop(index=drop_ids, inplace=True)
    print('Merged split skeletons and dropped', orig_row_count - len(df), 'rows')

    # Interpolate missing skeletons
    # Write directly to file to save memory; otherwise this step often crashes
    coord_columns = ['frame_num'] + coord_columns  # Interpolate frame number as well
    first_pid = True
    with open(out_fname, 'w', newline='', encoding='utf8') as ofile:
        for pid, pid_df in tqdm(df.groupby('person_id'), 'Interpolating',
                                total=df.person_id.nunique()):
            # Use Pandas built-in interpolate by constructing a DataFrame for this person that
            # includes NaNs where appropriate for the rows to be interpolated
            if pid_df.frame_num.max() - pid_df.frame_num.min() == len(pid_df) - 1:
                # Nothing to interpolate
                new_df = pid_df
            else:
                new_df = pd.DataFrame(
                    index=np.arange(pid_df.frame_num.min(), pid_df.frame_num.max() + 1))
                for col in pid_df.columns:  # Copy over column structure
                    new_df.insert(len(new_df.columns), col, pd.Series(dtype=pid_df[col].dtype))
                new_df.loc[pid_df.frame_num] = pid_df.values  # Copy existing data
                new_df['person_id'] = pid  # Fill in any gaps
                # Interpolate only the frame number and position columns
                new_df[coord_columns] = new_df[coord_columns].interpolate()
                assert not new_df[coord_columns].isna().any(None), 'Skeleton interpolation failed'
            if first_pid:  # Write header for first PID only
                new_df.to_csv(ofile, index=False)
                first_pid = False
            else:
                new_df.to_csv(ofile, index=False, header=False)


def track_file(fname, region, lookback, out_fname):
    # Load data and apply the first step: forming a rough linked list of closest distances
    df = pd.read_csv(fname)
    keypoints = [c.replace('_x', '') for c in df.columns if re.search(r'^keypoint\d*_x', c)]
    # If using a specific region, use only keypoints from that region; all keypoints from the
    # skeleton must be in this region
    if region:
        orig_row_count = len(df)
        x1, y1, x2, y2 = region
        for kp in keypoints:
            df = df[((df[kp + '_x'] == 0) | ((df[kp + '_x'] > x1) & (df[kp + '_x'] < x2))) &
                    ((df[kp + '_y'] == 0) | ((df[kp + '_y'] > y1) & (df[kp + '_y'] < y2)))]
        print('Selected', len(df), 'of', orig_row_count, 'rows from region:', region)
        global regionvals
        regionvals = region
    # TODO: Restrict keypoints to only certain ones for speed/accuracy improvement
    keypoint_dfs = {c: df[['frame_num', c + '_x', c + '_y', c + '_conf']] for c in keypoints}
    dist_df = pd.DataFrame({'frame_num': df.frame_num})
    for kp, kp_df in keypoint_dfs.items():
        kp_dist_df = calculate_distances(kp_df, kp, lookback)
        dist_df[kp_dist_df.columns] = kp_dist_df

    # Assign person IDs
    ids_df = assign_person_ids(dist_df)
    # Postprocess the ID assignment a bit further and save to output file
    postprocess_ids(ids_df, df, out_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script applies OpenPose and tracks individuals in the OpenPose output to '
        'assign person IDs to them.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dir', help='Process a directory of video files')
    group.add_argument('--file', help='Process one video file')
    parser.add_argument('--only_openpose', action='store_true',
                        help='Apply OpenPose only, do not track')
    parser.add_argument('--only_track', action='store_true',
                        help='Apply tracking to OpenPose output')
    parser.add_argument('--region', nargs=4, type=int,
                        metavar=('X_LEFT', 'Y_TOP', 'X_RIGHT', 'Y_BOTTOM'),
                        help='Only do tracking in a region of the video; the person must be '
                             'entirely in the given region to be tracked')
    parser.add_argument('--lookback', type=int, default=5,
                        help='Look back up to this many frames to find matches (default 5)')
    parser.add_argument('--outputfolder', type=str, default='',
                        help='Folder destination (relative or absolute path) to save the output '
                        '(default is current directory)')
    args = parser.parse_args()

    if args.file:
        if not args.only_track:  # Do OpenPose
            raise NotImplementedError('OpenPose tracking not yet implemented')
        if not args.only_openpose:  # Do tracking
            if args.region:  # Do tracking while restricting to user defined region
                out_fname = os.path.join(args.outputfolder, os.path.basename(args.file) +
                                         '-tracked-region-' + str(regionvals) + '.csv')
            else:  # Do tracking without restricting region
                out_fname = os.path.join(args.outputfolder, os.path.basename(args.file) +
                                         '-tracked.csv')
            track_file(args.file, args.region, args.lookback, out_fname)
        else:
            raise NotImplementedError('--dir functionality not yet implemented')
