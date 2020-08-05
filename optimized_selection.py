import pandas as pd
import numpy as np
from pulp import *
from scipy.stats import rankdata

class SelectPlayingTeam:

    def __init__(self, team_points, constconfig, colconfig):
        self.team_points = team_points
        self.constconfig = constconfig
        self.playernamecol = colconfig['PLAYERNAME']
        self.pointscol = colconfig['ACTUALPOINTS']
        self.predpointscol = colconfig['PREDPOINTS']
        self.playingrole = colconfig['PLAYINGROLE']
        self.playercost = colconfig['PLAYERCOST']
        self.predselection = colconfig['PREDSELECTION']
        self.actualselection = colconfig['ACTUALSELECTION']
        self.predselectionrank = colconfig['PREDSELECTIONRANK']
        self.actualselectionrank = colconfig['ACTUALSELECTIONRANK']

        return

    def select_top11_players(self):
        """
        function to select top 11 players out of the 22 players based on a given score
        :param input_df: dataset with the players name and match id
        :param predpointscol: column used to prioritize and pick top 11 players
        :param pointscol: actual points obtained by the players in the match
        :return: output_df: input_df wuth additional column with pred_selection_true
        """

        selected_team_pred = self.team_points.fillna(0).groupby('matchid').apply(self.get_optimized_team,
                                                                                 predpointscol=self.predpointscol,
                                                                                 selectioncol=self.predselection,
                                                                                 rankcol=self.predselectionrank)
        selected_team_actual = self.team_points.fillna(0).groupby('matchid').apply(self.get_optimized_team,
                                                                            predpointscol=self.pointscol,
                                                                            selectioncol=self.actualselection,
                                                                            rankcol=self.actualselectionrank)
        self.team_points = pd.merge(self.team_points, selected_team_pred, on=['matchid', 'playername'], how='left')
        self.team_points = pd.merge(self.team_points, selected_team_actual, on=['matchid', 'playername'], how='left')
        return

    def get_optimized_team(self, team_df, predpointscol, selectioncol, rankcol):
        """
        function to select the 11 people from the squad of the two teams playing a particular match
        :param team_df: dataframe with the actual and predicted points per player in a particular match
        :param pointscol: column name which has the points value
        :param selectioncol: column name which has the cost of each player
        :param rankcol: column name which has the cost of each player
        :return
        """
        player_list = list(team_df[self.playernamecol])
        points = dict(zip(player_list, team_df[predpointscol]))
        costs = dict(zip(player_list, team_df[self.playercost]))
        batsmen = dict(zip(player_list, np.where(team_df[self.playingrole] == 'Batsmen', 1, 0)))
        bowler = dict(zip(player_list, np.where(team_df[self.playingrole] == 'Bowler', 1, 0)))
        allrounder = dict(zip(player_list, np.where(team_df[self.playingrole] == 'AllRounder', 1, 0)))

        prob = LpProblem("Maximize Dream 11 Points", LpMaximize)

        player_vars = LpVariable.dicts("player", player_list, 0, 1, cat='Integer')

        prob += lpSum([points[i] * player_vars[i] for i in player_list]), "Total points earned by the team"

        prob += lpSum([costs[i] * player_vars[i] for i in player_list]) <= self.constconfig['MAXCOSTPOINT'], "MaximumCost"

        prob += lpSum([batsmen[i] * player_vars[i] for i in player_list]) >= self.constconfig['MINBATSMEN'], "MinimumBatsmen"
        prob += lpSum([batsmen[i] * player_vars[i] for i in player_list]) <= self.constconfig['MAXBATSMEN'], "MaximumBatsmen"

        prob += lpSum([bowler[i] * player_vars[i] for i in player_list]) >= self.constconfig['MINBOWLER'], "MinimumBowler"
        prob += lpSum([bowler[i] * player_vars[i] for i in player_list]) <= self.constconfig['MAXBOWLER'], "MaximumBowler"

        prob += lpSum([allrounder[i] * player_vars[i] for i in player_list]) >= self.constconfig['MINALLROUNDER'], "MinimumAllRounder"
        prob += lpSum([allrounder[i] * player_vars[i] for i in player_list]) <= self.constconfig['MAXALLROUNDER'], "MaximumAllRounder"

        # The problem is solved using PuLP's choice of Solver
        prob.solve()

        # The status of the solution is printed to the screen
        # print("Status:", LpStatus[prob.status])
        # print("The total points scored by the team is : ${}".format(value(prob.objective)))
        if value(prob.objective) != None:
            playernamelist = []
            playerscorelist = []

            for v in prob.variables():
                if v.varValue > 0:
                    a = v.name.replace("player_", "").replace("_", " ")
                    playernamelist.append(a)
                    playerscorelist.append(points[a])
            playerranklist = len(playerscorelist) - rankdata(playerscorelist, method='ordinal').astype(int)
            print(playernamelist)
            print(playerscorelist)
            print(playerranklist)
            playername = pd.Series(playernamelist, name='playername')
            playerrank = pd.Series(playerranklist, name=rankcol)
            selection_series = pd.Series(np.repeat(1, len(playername)), name=selectioncol)
            playerselec_df = pd.concat([playername, playerrank, selection_series], axis=1)
            return playerselec_df
        return


class RewardEstimate:

    def __init__(self, input_df):
        self.input_df = input_df
        return

    def compare_pred_vs_actual_points(self, minplayercount):
        """
        function to calculate the selected team's points achieved as a percentage of maximum possible
        :param input_df: input dataframe with predicted points per player and actual points scored
        :return: total_match_points
        """
        output_df = self.input_df
        output_df['pred_points_player'] = np.where(np.isnan(self.input_df['pred_selection_true']), np.nan, self.input_df['pred_selection_true']*self.input_df['total_points'])
        output_df['actual_points_player'] = np.where(np.isnan(self.input_df['actual_selection_true']), np.nan, self.input_df['actual_selection_true']*self.input_df['total_points'])
        total_match_points_pred = pd.DataFrame(output_df.groupby('matchid')['pred_points_player'].sum())
        total_match_points_actual = pd.DataFrame(output_df.groupby('matchid')['actual_points_player'].sum())
        count_player_selected = pd.DataFrame(output_df.groupby('matchid')['pred_selection_true'].sum()).rename(columns={'pred_selection_true':'pred_selection_cnt'})
        total_match_points = total_match_points_pred.merge(total_match_points_actual, left_index=True, right_index=True, how='left')
        total_match_points = total_match_points.merge(count_player_selected, left_index=True, right_index=True, how='left')
        total_match_points = total_match_points.reset_index()
        total_match_points['accuracy'] = np.where(total_match_points['pred_selection_cnt'] >= minplayercount,
                                                  (total_match_points['actual_points_player'] - total_match_points['pred_points_player'])/total_match_points['actual_points_player'], np.nan)
        self.total_match_points = total_match_points

        return


    def get_estimated_rewards(self, config, fixed_multipler) -> np.array:
        """
        function to calculate the expected rewards based on the estimated points
        :param input_df: input_df with the accuracy columns to estimate the rewards
        :return: output_df: input_df with the rewards earned column added
        """
        accuracy_series = self.total_match_points['accuracy']
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

        self.total_match_points['rewards_earned'] = np.select(conditions, choices, default=-1*fixed_multipler)

        return


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

def adjust_points_for_captaincy(input_df, colconfig):
    """
    function to adjust player's points if it is highest and 2nd highest
    :param input_df:
    :return:
    """
    output_df = input_df.copy()
    CAP_BUMP = 2.0
    VICE_CAP_BUMP = 1.5
    predpointscol = colconfig['PREDPOINTS']
    pointscol = colconfig['ACTUALPOINTS']
    predselectionrank = colconfig['PREDSELECTIONRANK']
    actualselectionrank = colconfig['ACTUALSELECTIONRANK']
    output_df[predpointscol] = np.where(output_df[predselectionrank] == 0, output_df[predpointscol] * CAP_BUMP,
                                        np.where(output_df[predselectionrank] == 1, output_df[predpointscol] * VICE_CAP_BUMP, output_df[predpointscol]))
    output_df[pointscol] = np.where(output_df[actualselectionrank] == 0, output_df[pointscol] * CAP_BUMP,
                                        np.where(output_df[actualselectionrank] == 1, output_df[pointscol] * VICE_CAP_BUMP, output_df[pointscol]))
    return output_df
