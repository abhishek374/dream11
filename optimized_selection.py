import pandas as pd
import numpy as np


def get_points_moving_avg(ipl_points, rolling_avg_window) -> pd.DataFrame:
    """
    function to calculate predicted points per player based on moving average method
    :param ipl_points: dataframe with the historical points achieved by the player
    :param rolling_avg_window: number of matches to take the rolling average over
    :return: ipl_points: dataframe with the columns total_points_avg, metric used to select the eventual team
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


def select_top11_players(input_df, predpointscol,pointscol,teamcount):
    """
    function to select top 11 players out of the 22 players based on a given score
    :param input_df: dataset with the players name and match id
    :param predpointscol: column used to prioritize and pick top 11 players
    :param pointscol: actual points obtained by the players in the match
    :return: output_df: input_df wuth additional column with pred_selection_true

    """
    # TODO Add the module to check for constraint within select11

    output_df = input_df.copy()
    output_df['pred_selection_rank'] = input_df.groupby('matchid')[predpointscol].rank(ascending=False, method='first')
    output_df['pred_selection_true'] = np.where(np.isnan(output_df['pred_selection_rank']), 0, np.where((output_df['pred_selection_rank'] <= 11), 1, 0))
    output_df['selection_rank'] = input_df.groupby('matchid')[pointscol].rank(ascending=False, method='first')
    output_df['selection_true'] = np.where(np.isnan(output_df['selection_rank']), 0, np.where((output_df['selection_rank'] <= 11), 1, 0))
    return output_df


def adjust_points_for_captaincy(input_df, predpointscol, pointscol, playercount):
    """
    function to adjust player's points if it is highest and 2nd highest
    :param input_df:
    :return:
    """
    output_df = input_df.copy()
    CAP_BUMP = 2.0
    VICE_CAP_BUMP = 1.5
    output_df[predpointscol] = np.where(output_df['pred_selection_rank'] == 1, output_df[predpointscol] * CAP_BUMP,
                                        np.where(output_df['pred_selection_rank'] == 2, output_df[predpointscol] * VICE_CAP_BUMP, output_df[predpointscol]))
    output_df[pointscol] = np.where(output_df['selection_rank'] == 1, output_df[pointscol] * CAP_BUMP,
                                        np.where(output_df['selection_rank'] == 2, output_df[pointscol] * VICE_CAP_BUMP, output_df[pointscol]))
    return output_df


def compare_pred_vs_actual_points(input_df) -> np.array:
    """
    function to calculate the selected team's points achieved as a percentage of maximum possible
    :param input_df: input dataframe with predicted points per player and actual points scored
    :return:
    """
    output_df = input_df.copy()
    output_df['pred_points_player'] = np.where(np.isnan(input_df['pred_selection_true']), np.nan, input_df['pred_selection_true']*input_df['total_points'])
    output_df['actual_points_player'] = np.where(np.isnan(input_df['selection_true']), np.nan, input_df['selection_true']*input_df['total_points'])
    total_match_points_pred = pd.DataFrame(output_df.groupby('matchid')['pred_points_player'].sum())
    total_match_points_actual = pd.DataFrame(output_df.groupby('matchid')['actual_points_player'].sum())
    count_player_selected = pd.DataFrame(output_df.groupby('matchid')['pred_selection_true'].sum()).rename(columns={'pred_selection_true':'pred_selection_cnt'})
    total_match_points = total_match_points_pred.merge(total_match_points_actual, left_index=True, right_index=True, how='left')
    total_match_points = total_match_points.merge(count_player_selected, left_index=True, right_index=True, how='left')
    total_match_points = total_match_points.reset_index()
    total_match_points['accuracy'] = np.where(total_match_points['pred_selection_cnt'] >= 10,
                                              (total_match_points['actual_points_player'] - total_match_points['pred_points_player'])/total_match_points['actual_points_player'], np.nan)
    return total_match_points


def get_estimated_rewards(input_df, config, fixed_multipler) -> np.array:
    """
    function to calculate the expected rewards based on the estimated points
    :param input_df: input_df with the accuracy columns to estimate the rewards
    :return: output_df: input_df with the rewards earned column added
    """
    output_df = input_df.copy()
    accuracy_series = input_df['accuracy']
    conditions = [(np.isnan(accuracy_series)),
                  (accuracy_series < .01),
                  (accuracy_series < 0.02),
                  (accuracy_series < 0.03),
                  (accuracy_series < 0.04),
                  (accuracy_series < 0.05),
                  (accuracy_series < 0.06),
                  (accuracy_series < 0.08),
                  (accuracy_series < 0.10),
                  (accuracy_series < 0.15),
                  (accuracy_series < 0.20),
                  (accuracy_series < 0.25)]

    choices = [0,
               (config['1per'] - 1) * fixed_multipler,
               (config['2per'] - 1) * fixed_multipler,
               (config['3per'] - 1) * fixed_multipler,
               (config['4per'] - 1) * fixed_multipler,
               (config['5per'] - 1) * fixed_multipler,
               (config['6per'] - 1) * fixed_multipler,
               (config['8per'] - 1) * fixed_multipler,
               (config['10per'] - 1) * fixed_multipler,
               (config['15per'] - 1) * fixed_multipler,
               (config['20per'] - 1) * fixed_multipler,
               (config['25per'] - 1) * fixed_multipler]

    output_df['rewards_earned'] = np.select(conditions, choices, default=-1*fixed_multipler)
    return output_df


def check_dream11_constraint(team_df):
    """
    function to measure the constraints of number of players from each type of playing role that can be selected
    :param team_df: dataframe of the team with the cost and playing role of each player
    :return: Check: a boolean data type clearing a particular team choice
    """
    summary_df = pd.DataFrame(team_df.groupby(['playing_role'])['playername'].count()).reset_index()
    print(summary_df)
    if ((summary_df[summary_df['playing_role'] == 'All Rounder']['playername'] > 1) &
            (summary_df[summary_df['playing_role'] == 'All Rounder']['playername'] < 4)):
        if summary_df[summary_df['playing_role'] == 'Batsmen']['playername'] > 1:
            if summary_df[summary_df['playing_role'] == 'Bowler']['playername'] > 1:
                Check = True
            else:
                Check = False
        else:
            Check = False
    else:
        Check = False


