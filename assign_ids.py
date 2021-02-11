# TODO: Probably eventually incorporate this into tracking_analysis.py, but for now it is convenient
# to have it separate for the sake of developing individual pieces without re-running everything.
import pandas as pd


df = pd.read_csv('14_11_10_Green_20to21min.csv-match_indices.csv')
keypoints = [c.replace('_closest_dist', '') for c in df.columns if c.endswith('_closest_dist')]
for row_i, row in df.iterrows():
    # Cols to consider: keypoint1_closest_index,keypoint1_closest_dist,keypoint1_second_closest_dist
    # Select keypoints that were good enough matches to participate in the overall match decision
    # Actually, maybe bad, non-zero matches should be evidence that the skeletong does NOT match
    # TODO: Also need to prevent tracking_analysis matches with 0,0 non-matches in prev. frames
    voting_kps = [k + '_closest_index' for k in keypoints
                  if row[k + '_closest_dist'] < 10 and row[k + '_second_closest_dist'] > 10]
    # TODO: The problem with matching on index 52 is that the skeleton is broken up into 3 parts on
    # indices 34, 44, and 49, all of which are good matches because each is a real match.
    #   -- maybe merge skeletons somehow? Only if there are no conflicting matches?
    print(row[voting_kps].value_counts())
