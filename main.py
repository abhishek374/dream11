from data_prep import ScoreCard
import pandas as pd
from optimized_selection import  *
# reading the source file from local
matchdata = pd.read_csv(r'matchdata.csv')

# points as per dream11 website
pointsconfig = {'total_runs': 1,
                'run_6': 2,
                'run_4': 1,
                  '>=50': 8,
                  '>=100': 16,
                  'duck': -2,
                  'total_wickets': 25,
                  '>=4W': 8,
                  '>=5W': 16,
                  'maiden_overs': 8,
                  '<=4E': 6,
                  '<5E': 4,
                  '<6E': 2,
                  '>9E': -2,
                  '>10E': -4,
                  '>11E': -6
                }


# getting the scorecard from a batsmen's perspective
ipl_scorecard = ScoreCard(matchdata.copy())
# getting the points from a batsmen's perspective
batting_points = ipl_scorecard.get_batting_points(pointsconfig)
# getting the points from a bowler's perspective
bowling_points = ipl_scorecard.get_bowling_points(pointsconfig)
# merging both the batsmen and bowler's points to get a single view
ipl_scorecard_points = ipl_scorecard.merge_batsmen_bowler_scorecard(batting_points, bowling_points)
# writing
ipl_scorecard_points.to_csv(r'ipl_scorecard_points.csv', index=False)

# Defining the metric to select the players
ipl_scorecard_points_avg = get_points_moving_avg(ipl_scorecard_points.copy(), rolling_avg_window=10)
# selecting the 11 players from a team of 22 based on historic points average

ipl_scorecard_points = select_top11_players(ipl_scorecard_points_avg, 'total_points_avg', 'total_points')
accuracy_df = compare_pred_vs_actual_points(ipl_scorecard_points)
rewards_array = get_estimated_rewards(accuracy_df, rewardconfig, fixed_multipler=50)
accuracy_df.to_csv(r'accuracy_df.csv', index=False)
