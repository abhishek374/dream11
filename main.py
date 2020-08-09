from data_prep import ScoreCard, Dream11Points, FeatEngineering
from optimized_selection import *
from point_prediction import PointPred
import pandas as pd
# reading the source file from local
matchdata = pd.read_csv(r'Data/matchdata.csv')
matchsummary = pd.read_csv(r'Data/matchsummary.csv')
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

constconfig = {'MAXCOSTPOINT': 110,
               'MINBATSMEN': 3,
               'MAXBATSMEN': 7,
               'MINBOWLER': 3,
               'MAXBOWLER': 6,
               'MINALLROUNDER': 1,
               'MAXALLROUNDER': 4}

colconfig = {'MATCHID': 'matchid',
             'BATSMANNAME': 'batsmanname',
             'BOWLERNAME': 'bowlername',
             'SCOREVALUE': 'scorevalue',
             'OVER': 'over',
             'INNINGS': 'innings',
             'VENUE':'venue',
             'BATTINGORDER': 'fallofwickets',
             'BATTINGTEAM': 'battingteam',
             'BOWLINGTEAM': 'bowlingteam',
             'PLAYERNAME': 'playername',
             'TOTALBATPOINTS':'total_bat_points',
             'TOTALBALLPOINTS':'total_bowl_points',
             'ACTUALPOINTS': 'total_points',
             'PREDPOINTS': 'total_points_avg10',
             'PLAYERCOST': 'playercost',
             'PLAYINGROLE': 'playing_role',
             'PREDSELECTION': 'pred_selection_true',
             'ACTUALSELECTION': 'actual_selection_true',
             'PREDSELECTIONRANK': 'pred_selection_rank',
             'ACTUALSELECTIONRANK': 'actual_selection_rank'}

# getting the scorecard from a batsmen's perspective
ipl_scorecard = ScoreCard(matchdata.copy())
# merging both the batsmen and bowler's points to get a single view
ipl_scorecard.merge_player_scorecard()
# calculating the points scored by the players based on dream11 scoring method
ipl_scorecard_points = Dream11Points(ipl_scorecard.ipl_merged_scorecard, pointsconfig)
ipl_scorecard_points.get_batsmen_bowler_points()
# writing the scorecard to ipl_scorecard_points.csv
# ipl_scorecard_points.player_scorecard.to_csv(r'Data/ipl_scorecard_points.csv', index=False)
##################Part to create additional features for modeling#######################################################
ROLLINGWINDOW = 10

FeatEng = FeatEngineering(ipl_scorecard_points.player_scorecard, matchsummary)
FeatEng.add_venue_info()
FeatEng.add_homegame_flag()
FeatEng.add_toss_info()
FeatEng.add_player_match_count()
FeatEng.add_lagging_feat(colconfig['MATCHID'], colconfig['VENUE'], 10, colconfig['TOTALBATPOINTS'], colconfig['TOTALBALLPOINTS'])
FeatEng.add_lagging_feat(colconfig['MATCHID'], colconfig['PLAYERNAME'], 10, colconfig['ACTUALPOINTS'], colconfig['TOTALBATPOINTS'], colconfig['TOTALBALLPOINTS'], colconfig['BATTINGORDER'], 'total_balls_bowled')

#ipl_scorecard_points_avg = PointPred().get_points_moving_avg(ipl_scorecard_points.player_scorecard, rolling_avg_window=ROLLINGWINDOW)

# writing the scorecard to save it
FeatEng.ipl_features.to_csv(r'Data/ipl_scorecard_points_featengg.csv', index=False)

##################Part to run the optimization to select the playing 11#################################################
# selecting the 11 players from a team of 22 based on historic points average
# Temp till we get better alternate to cost of each player
FeatEng.ipl_features['playercost'] = 10
SQUADCOUNT = 11
TOTALPLAYERCOUNT = 22

# get the team by running binary LP solver
optimum_team = SelectPlayingTeam(FeatEng.ipl_features, constconfig, colconfig)
# select Top11 based on the predicted points
optimum_team.select_top11_players(pointscol=colconfig['PREDPOINTS'], selectioncol=colconfig['PREDSELECTION'],
                                  rankcol=colconfig['PREDSELECTIONRANK'], adjustcappoints=True)
##################Part to calculate the accuracy of the selected 11 if the actual 11 is available#######################
# variable to control if we want to compare with the actual Data
ACTUALDATA = True

if ACTUALDATA:
    # select Top11 based on the actual points
    optimum_team.select_top11_players(pointscol=colconfig['ACTUALPOINTS'], selectioncol=colconfig['ACTUALSELECTION'],
                                      rankcol=colconfig['ACTUALSELECTIONRANK'], adjustcappoints=True)

    optimum_team.team_points.to_csv(r'Data/team_points.csv')
    # get the rewards estimate
    ipl_team_rewards = RewardEstimate(optimum_team.team_points, matchdata.copy())

    # get the percentile of the predicted team vs actual tam
    ipl_team_rewards.compare_pred_vs_actual_points(minplayercount=SQUADCOUNT)

    # estimating the monetary impact of the project
    ipl_team_rewards.get_estimated_rewards(rewardconfig, fixed_multipler=50)
    ipl_team_rewards.total_match_points.to_csv(r'Data/rewards_df.csv', index=False)

    # calculating a yearly summary of the model
    yearly_summary = ipl_team_rewards.get_rewards_summary()
    yearly_summary.to_csv(r'Data/rewards_yearly_summary.csv', index=False)
    print(ipl_team_rewards.total_match_points['rewards_earned'].sum())

#TODO make the constraint for allrounder, batsmen, bowler
