import pandas as pd
import numpy as np

ipl_scorecard_points = pd.read_csv(r'ipl_scorecard_points.csv')
def get_avg_point(ipl_points, rolling_avg_window):
    """
    function to get the average points scored by a player

    :return:
    """
    ipl_points = ipl_points.sort_values(by=['matchid', 'playername'], ascending=True)
    ipl_points.set_index('matchid', inplace=True)
    player_avg_points = pd.DataFrame(ipl_points.groupby(['playername'])['total_points'].rolling(rolling_avg_window).mean()).reset_index(). \
        rename(columns={'total_points': 'total_points_avg'})
    player_avg_points = player_avg_points.sort_values(by=['matchid', 'playername'], ascending=True)
    player_avg_points.set_index('matchid', inplace=True)
    player_avg_points['total_points_avg'] = pd.DataFrame(player_avg_points.groupby(['playername'])['total_points_avg'].shift(1))
    ipl_points.reset_index(inplace=True)
    player_avg_points.reset_index(inplace=True)
    ipl_points = pd.merge(ipl_points, player_avg_points, on=['matchid', 'playername'], how='left')
    return ipl_points

def select_top11_players(master_df, predpointscol,pointscol):
    """
    function to select top 11 players out of the 22 players based on a given score
    master_df: dataset with the players name and match id
    pointscol: column used to prioritize and pick top 11 players
    :return:

    """
    master_df['pred_selection_rank'] = master_df.groupby('matchid')[predpointscol].rank(ascending=False)
    master_df['pred_selection_true'] = np.where(master_df['pred_selection_rank'] >11, 0,1)

    master_df['selection_rank'] = master_df.groupby('matchid')[pointscol].rank(ascending=False)
    master_df['selection_true'] = np.where(master_df['selection_rank'] >11, 0,1)
    return master_df

def compare_pred_vs_actual_points(input_df):
    """

    :param input_df: input dataframe with predicted points per player and actual points scored
    :return:
    """
    input_df['prep_points_player'] = input_df['pred_selection_true']*input_df['total_points']
    input_df['actual_points_player'] = input_df['selection_true'] * input_df['total_points']
    total_match_points_pred = input_df.groupby('matchid')['prep_points_player'].sum()
    total_match_points_actual = input_df.groupby('matchid')['total_points'].sum()
    accuracy = (total_match_points_actual - total_match_points_pred)/ total_match_points_actual
    return accuracy
ipl_scorecard_points_avg = get_avg_point(ipl_scorecard_points.copy(), rolling_avg_window=10)
ipl_scorecard_points = select_top11_players(ipl_scorecard_points_avg, 'total_points_avg','total_points')
accuracy_array = compare_pred_vs_actual_points(ipl_scorecard_points)
accuracy_array.to_csv(r'accuracy_array.csv', index=False)

# TODO Add the module to check for constraint
# TODO Define a player as batsmen, bowler or all rounder
# TODO ROI Calculator

