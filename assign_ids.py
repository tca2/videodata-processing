# TODO: Probably eventually incorporate this into tracking_analysis.py, but for now it is convenient
# to have it separate for the sake of developing individual pieces without re-running everything.
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


df = pd.read_csv('14_11_10_Green_20to21min.csv-match_indices.csv')
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
    # TODO: Do a follow-up pass to merge skeletons in the same frame with the same person_id

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
df.to_csv('14_11_10_Green_20to21min.csv-match_indices.csv-ids.csv', index=False)
