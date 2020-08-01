import pandas as pd
import numpy as np

ipl_scorecard_points = pd.read_csv(r'ipl_scorecard_points.csv')
def get_points_moving_avg(ipl_points, rolling_avg_window):
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

def select_top11_players(input_df, predpointscol,pointscol):
    """
    function to select top 11 players out of the 22 players based on a given score
    master_df: dataset with the players name and match id
    pointscol: column used to prioritize and pick top 11 players
    :return:

    """
    output_df = input_df.copy()
    output_df['pred_selection_rank'] = input_df.groupby('matchid')[predpointscol].rank(ascending=False)
    output_df['pred_selection_true'] = np.where(np.isnan(output_df['pred_selection_rank']), 0,
                                               np.where((output_df['pred_selection_rank'] < 11), 1, 0))
    output_df['selection_rank'] = input_df.groupby('matchid')[pointscol].rank(ascending=False)
    output_df['selection_true'] = np.where(np.isnan(output_df['selection_rank']), 0,
                                               np.where((output_df['selection_rank'] < 11), 1, 0))
    return output_df

def compare_pred_vs_actual_points(input_df) -> np.array:
    """

    input_df: input dataframe with predicted points per player and actual points scored
    :return:
    """
    output_df = input_df.copy()
    output_df['pred_points_player'] = np.where(np.isnan(input_df['pred_selection_true']), np.nan, input_df['pred_selection_true']*input_df['total_points'])
    output_df['actual_points_player'] = np.where(np.isnan(input_df['selection_true']), np.nan, input_df['selection_true']*input_df['total_points'])

    total_match_points_pred = pd.DataFrame(output_df.groupby('matchid')['pred_points_player'].sum())
    total_match_points_actual = pd.DataFrame(output_df.groupby('matchid')['actual_points_player'].sum())

    count_player_selected = pd.DataFrame(output_df.groupby('matchid')['pred_selection_true'].sum()).rename(columns = {'pred_selection_true':'pred_selection_cnt'})

    total_match_points = total_match_points_pred.merge(total_match_points_actual, left_index=True, right_index=True, how='left')
    total_match_points = total_match_points.merge(count_player_selected, left_index=True, right_index=True, how='left')
    total_match_points = total_match_points.reset_index()
    total_match_points['accuracy'] = np.where(total_match_points['pred_selection_cnt'] >= 10,
                                              (total_match_points['actual_points_player'] - total_match_points['pred_points_player'])/total_match_points['actual_points_player'], np.nan)
    return total_match_points

rewardconfig = {'1per': 1000,
                '2per': 200,
                '3per': 100,
                '4per': 80,
                '5per': 40,
                '6per': 20,
                '8per': 5,
                '10per': 2}

def get_estimated_rewards(input_df, config, fixed_multipler) -> np.array:
    """
    function to calculate the expected rewards based on the estimated points
    input_df: input_df with the accuracy columns to estimate the rewards
    :return: rewardsarray
    """
    accuracy_series = input_df['accuracy']
    conditions = [(np.isnan(accuracy_series)),
                  (accuracy_series < .01),
                  (accuracy_series < 0.02),
                  (accuracy_series < 0.03),
                  (accuracy_series < 0.04),
                  (accuracy_series < 0.05),
                  (accuracy_series < 0.06),
                  (accuracy_series < 0.08),
                  (accuracy_series < 0.10)]

    choices = [0,
               (config['1per'] - 1) * fixed_multipler,
               (config['2per'] - 1) * fixed_multipler,
                (config['3per'] - 1) * fixed_multipler,
                 (config['4per'] - 1) * fixed_multipler,
                  (config['5per'] - 1) * fixed_multipler,
                   (config['6per'] - 1) * fixed_multipler,
                    (config['8per'] - 1) * fixed_multipler,
                     (config['10per'] - 1) * fixed_multipler]

    rewards_array = np.select(conditions, choices, default=-1*fixed_multipler)
    return rewards_array


# TODO Add the module to check for constraint
# TODO Define a player as batsmen, bowler or all rounder

