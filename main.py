from data_prep import ScoreCard, Dream11Points
from optimized_selection import *
import pandas as pd
# reading the source file from local
matchdata = pd.read_csv(r'matchdata.csv')

# points as per dream11 website
pointsconfig = {
                'total_runs': 1,
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
# rewards as per the result of one of the matches on dream11
rewardconfig = {
                '1per': 5000,#10000
                '2per': 3000,#6000
                '3per': 500,#500
                '4per': 200,
                '5per': 100,
                '6per': 80,
                '8per': 20,
                '10per': 8,
                '15per': 2.5,
                '20per': 2,
                '25per': 1
                }

# getting the scorecard from a batsmen's perspective
ipl_scorecard = ScoreCard(matchdata.copy())
ipl_scorecard_summary = ipl_scorecard.merge_player_scorecard()

# merging both the batsmen and bowler's points to get a single view
ipl_scorecard_points = Dream11Points(ipl_scorecard_summary, pointsconfig).get_batsmen_bowler_points()

# Defining the metric to select the players
ROLLINGWINDOW = 10
ipl_scorecard_points_avg = get_points_moving_avg(ipl_scorecard_points.copy(), rolling_avg_window=ROLLINGWINDOW)

# writing the scorecard to save it
ipl_scorecard_points_avg.to_csv(r'ipl_scorecard_points_avg.csv', index=False)

# selecting the 11 players from a team of 22 based on historic points average
TEAMCOUNT = 11
ipl_scorecard_points = select_top11_players(ipl_scorecard_points_avg, 'total_points_avg', 'total_points', teamcount=TEAMCOUNT)
# calculating the accuracy of the prediction against the maximum possible in the match
accuracy_df = compare_pred_vs_actual_points(ipl_scorecard_points)
# estimating the monetary impact of the project
rewards_df = get_estimated_rewards(accuracy_df, rewardconfig, fixed_multipler=50)
rewards_df.to_csv(r'rewards_df.csv', index=False)
print(rewards_df['rewards_earned'].sum())

# TODO Add dream11 playing role constraint to the select11 function
# TODO Enable captain and vice captain role in point calculation

