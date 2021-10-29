#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import re
from collections import deque, defaultdict

import pandas as pd
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm


# Calculate closest distance, closest distance owner, second closest distance
def calculate_distances(keypoint_df, keypoint_num):
    matches = pd.DataFrame(index=keypoint_df.index)
    matches['frame_num'] = keypoint_df.frame_num
    # Count low-confidence detections as non-detections because they can be misleading
    matches[keypoint_num + '_detected'] = \
        (keypoint_df[keypoint_num + '_conf'] > .3).astype(int)
    prev_frames = deque(maxlen=5)  # Number of frames into the past to search for matches TODO: Make an arg
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
                if closest_dist < 15:  # TODO: Autodetect this or something
                    break  # Close enough; stop looking further back
            matches.at[row_i, [keypoint_num + '_closest_index', keypoint_num + '_closest_dist',
                               keypoint_num + '_second_closest_dist']] = \
                [index_closest, closest_dist, second_closest_dist]
        prev_frames.append(frame)
    return matches


# Based on a set of closest/second-closest distances, try to assign person IDs
def assign_person_ids(matches_df, keypoints):
    df = matches_df.copy()
    # Precalculate sets of detected keypoints to improve speed
    detected_kps = {i: set([k for k in keypoints if r[k + '_detected']])
                    for i, r in tqdm(df.iterrows(), 'Counting detected keypoints', total=len(df))}
    # Go through frames in reverse order to follow links to earlier frames. Then, if we reach a
    # frame that has no frames linked to it, assign a new person ID. Alternatively, if we reach a
    # frame that has multiple frames linked to it, resolve the ambiguity via voting for the most
    # keypoints matched.
    proposed_ids = defaultdict(set)
    df.insert(2, 'person_id', '')
    df.insert(3, 'new_id', 0)
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
            df.at[row_i, ['person_id', 'new_id']] = ['idx' + str(row_i), 1]

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


def postprocess_ids(ids_df, orig_df, keypoints):
    df = ids_df.copy()
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
    interpolated_dfs = []
    for _, pid_df in tqdm(df.groupby('person_id'), 'Interpolating', total=df.person_id.nunique()):
        # Use Pandas built-in interpolate by constructing a DataFrame for this person that includes
        # NaNs where appropriate for the rows to be interpolated
        if pid_df.frame_num.max() - pid_df.frame_num.min() == len(pid_df) - 1:
            # Nothing to interpolate
            interpolated_dfs.append(pid_df)
        else:
            new_df = pd.DataFrame(
                index=np.arange(pid_df.frame_num.min(), pid_df.frame_num.max() + 1))
            for col in pid_df.columns:
                new_df.insert(len(new_df.columns), col, pd.Series(dtype=pid_df[col].dtype))
            new_df.loc[pid_df.frame_num] = pid_df.values
            # Interpolate only the position columns we care about
            new_df[coord_columns] = new_df[coord_columns].interpolate()
            assert not new_df[coord_columns].isna().any().any(), 'Interpolating skeletons failed'
            interpolated_dfs.append(new_df)

    # Merge interpolated values to form combined DataFrame
    return pd.concat(interpolated_dfs)


def track_file(fname):
    # Load data and apply the first step: forming a rough linked list of closest distances
    df = pd.read_csv(fname)
    # TODO: Restrict keypoints to only certain ones for speed/accuracy improvement
    keypoints = [c.replace('_x', '') for c in df.columns if re.search(r'^keypoint\d*_x', c)]
    keypoint_dfs = {c: df[['frame_num', c + '_x', c + '_y', c + '_conf']] for c in keypoints}
    dist_df = pd.DataFrame(index=df.index)
    for key, value in keypoint_dfs.items():
        kp_dist_df = calculate_distances(value, key)
        dist_df[kp_dist_df.columns] = kp_dist_df

    # Assign person IDs
    ids_df = assign_person_ids(dist_df, keypoints)
    # Postprocess the ID assignment a bit further
    return postprocess_ids(ids_df, df, keypoints)


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
    args = parser.parse_args()

    if args.file:
        if not args.only_track:  # Do OpenPose
            raise NotImplementedError('OpenPose tracking not yet implemented')
        if not args.only_openpose:  # Do tracking
            df = track_file(args.file)
            df.to_csv(args.file + '-tracked.csv', index=False)  # TODO: allow output folder as arg
    else:
        raise NotImplementedError('--dir functionality not yet implemented')
