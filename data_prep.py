import pandas as pd
import numpy as np

class ScoreCard:

    def __init__(self, matchdata):
        self.matchdata = matchdata
        self.matchdata['batsmanname'] = self.matchdata['batsmanname'].str.replace('-', ' ')
        self.matchdata['bowlername'] = self.matchdata['bowlername'].str.replace('-', ' ')
        self.matchdata['over_num'] = self.matchdata['over'].apply(lambda x: int(x))
        return

    def batsmen_summary_fun(self) -> None:
        """
        matchdata: pandas dataframe of ball by ball data for the matches
        return: batsmen_summary: pandas datarame with summary scorecard of a batsmen
        """
        batsmen_score = pd.DataFrame(self.matchdata.groupby(['matchid', 'batsmanname'])['scorevalue'].sum()).\
            rename(columns={"scorevalue": "total_runs"})
        batsmen_ball_faced = pd.DataFrame(self.matchdata.groupby(['matchid', 'batsmanname'])['over'].count()).\
            rename(columns={"over": "total_balls_faced"})
        batsmen_ball_faced_legal = pd.DataFrame(self.matchdata.groupby(['matchid', 'batsmanname'])['over'].nunique()).\
            rename(columns={"over": "total_legal_balls_faced"})
        batsmen_scores6 = pd.DataFrame(self.matchdata[self.matchdata['scorevalue'] == 6].groupby(['matchid', 'batsmanname'])['scorevalue'].count()).\
            rename(columns={"scorevalue": "run_6"})
        batsmen_scores4 = pd.DataFrame(self.matchdata[self.matchdata['scorevalue'] == 4].groupby(['matchid', 'batsmanname'])['scorevalue'].count()).\
            rename(columns={"scorevalue": "run_4"})
        batsmen_position = pd.DataFrame(self.matchdata.groupby(['matchid', 'batsmanname'])['fallofwickets'].min())
        df_list = [batsmen_score, batsmen_ball_faced, batsmen_ball_faced_legal, batsmen_scores6, batsmen_scores4, batsmen_position]
        batsmen_summary = pd.concat(df_list, join='outer', axis=1).fillna(np.nan).reset_index()
        batsmen_summary = pd.merge(batsmen_summary, self.matchdata[['matchid', 'batsmanname', 'innings', 'battingteam', 'bowlingteam']].
                                   drop_duplicates(), on=['matchid', 'batsmanname'], how='left')
        batsmen_summary.rename(columns={'innings': 'batsmen_innings', 'batsmanname': 'playername', 'battingteam': 'batsmen_battingteam',
                                        'bowlingteam': 'batsmen_bowlingteam'}, inplace=True)
        self.batsmen_summary = batsmen_summary
        return

    def bowler_summary_fun(self) -> None:
        """
        matchdata: pandas dataframe of ball by ball data for the matches

        return: bowler_summary: pandas datarame with summary scorecard of a bowler
        """
        bowler_wickets = pd.DataFrame(self.matchdata[(((self.matchdata['dismissal'] == 't') | (self.matchdata['dismissal'] == True)) & (~self.matchdata['dismissedtype'].isin(['run out','retired hurt'])))].
                                      groupby(['matchid', 'bowlername'])['dismissal'].count()).\
            rename(columns={"dismissal": "total_wickets"})
        bowler_overs_bow = pd.DataFrame(self.matchdata.groupby(['matchid', 'bowlername'])['over'].count()).\
            rename(columns={"over": "total_balls_bowled"})
        bowler_ball_faced_legal = pd.DataFrame(self.matchdata.groupby(['matchid', 'bowlername'])['over'].nunique()).\
            rename(columns={"over": "total_legal_balls_bowled"})
        # calculating the number of maiden overs bowled in a match
        bowler_runs_given = pd.DataFrame(self.matchdata.groupby(['matchid', 'bowlername'])['batsmanscorevalue'].sum()).\
            rename(columns={"batsmanscorevalue": "total_runs_given"})
        matchdata_runs_per_over = pd.DataFrame(self.matchdata.groupby(['matchid', 'bowlername', 'over_num'])['batsmanscorevalue'].sum()).reset_index()
        matchdata_runs_per_over['maiden_overs'] = np.where(matchdata_runs_per_over['batsmanscorevalue'] == 0, 1, 0)
        matchdata_runs_per_over = matchdata_runs_per_over.groupby(['matchid', 'bowlername'])['maiden_overs'].sum()
        # combining all the features created so far
        df_list = [bowler_wickets, bowler_overs_bow, bowler_ball_faced_legal, bowler_runs_given, matchdata_runs_per_over]
        bowler_summary = pd.concat(df_list, join='outer', axis=1).fillna(np.nan).reset_index()
        # adding additional columns just in case we need for modeling
        bowler_summary = pd.merge(bowler_summary, self.matchdata[['matchid', 'bowlername', 'innings', 'battingteam', 'bowlingteam']].drop_duplicates(), on=['matchid', 'bowlername'], how='left')
        bowler_summary.rename(columns={'innings': 'bowlers_innings', 'bowlername': 'playername', 'battingteam': 'bowler_battingteam',
                                       'bowlingteam': 'bowler_bowlingteam'}, inplace=True)
        bowler_summary['economy_rate'] = (bowler_summary['total_runs_given'] * 6) / bowler_summary['total_legal_balls_bowled']
        self.bowler_summary = bowler_summary
        return

    def merge_player_scorecard(self):
        """
        batting_points: scorecard from batsmen perspective with dream11 points
        bowling_points: scorecard from bowling perspective with dream11 points
        :return: ipl_points: merged dataset with the total points per player in a match
        """
        self.batsmen_summary_fun()
        self.bowler_summary_fun()
        ipl_merged_scorecard = pd.merge(self.batsmen_summary, self.bowler_summary, on=['matchid', 'playername'], how='outer')
        player_avg = self.get_player_role(ipl_merged_scorecard)
        ipl_merged_scorecard = pd.merge(ipl_merged_scorecard, player_avg[['playername', 'playing_role']], on='playername', how='left')
        self.ipl_merged_scorecard = ipl_merged_scorecard
        return

    def get_player_role(self, input_df) -> pd.DataFrame:
        """"
        function to get the players role in the team
        input_df: input dataframe that has the combined scorecard of the players with batting and bowling
        :return player_avg: df with the player name and player's playing role
        """
        MINAVGBALLSFACED = 8
        MINAVGBOWLSBOWLED = 6
        player_avg = input_df[['playername', 'total_balls_faced', 'total_balls_bowled']].fillna(0)
        player_avg = pd.DataFrame(player_avg.groupby('playername')[['total_balls_faced', 'total_balls_bowled']].mean())
        conditions = [((player_avg['total_balls_faced'] >= MINAVGBALLSFACED) & (player_avg['total_balls_bowled'] >= MINAVGBOWLSBOWLED)),
                      (player_avg['total_balls_bowled'] >= MINAVGBOWLSBOWLED)]
        choices = ['AllRounder', 'Bowler']
        player_avg['playing_role'] = np.select(conditions, choices, default='Batsmen')
        player_avg = player_avg.reset_index()
        return player_avg


class Dream11Points:

    def __init__(self, player_scorecard, pointsconfig):
        self.player_scorecard = player_scorecard
        self.pointsconfig = pointsconfig

        return

    def get_batting_points(self) -> None:
        """
        pointsconfig: dictionary with points per score type as per dream11
        :return:
        """
        self.player_scorecard['total_runs_points'] = self.player_scorecard['total_runs'] * self.pointsconfig['total_runs']
        self.player_scorecard['run_6_points'] = self.pointsconfig['run_6'] * self.player_scorecard['run_6']
        self.player_scorecard['run_4_points'] = self.pointsconfig['run_4'] * self.player_scorecard['run_4']
        self.player_scorecard['run_bonus_points'] = np.where(self.player_scorecard['total_runs'] == 0, self.pointsconfig['duck'], 0)
        self.player_scorecard['run_bonus_points'] = np.where(self.player_scorecard['total_runs'] >= 50, self.pointsconfig['>=50'] +
                                                         self.player_scorecard['run_bonus_points'], self.player_scorecard['run_bonus_points'])
        self.player_scorecard['run_bonus_points'] = np.where(self.player_scorecard['total_runs'] >= 100, self.pointsconfig['>=100'] +
                                                         self.player_scorecard['run_bonus_points'], self.player_scorecard['run_bonus_points'])
        self.player_scorecard['total_bat_points'] = np.nan
        self.player_scorecard['total_bat_points'] = self.player_scorecard['total_runs_points'].add(self.player_scorecard['run_6_points'], fill_value=0). \
            add(self.player_scorecard['run_4_points'], fill_value=0).add(self.player_scorecard['run_bonus_points'], fill_value=0)
        self.player_scorecard['total_bat_points'] = np.where(self.player_scorecard['total_balls_faced'] >= 1, self.player_scorecard['total_bat_points'], np.nan)

        return

    def get_bowling_points(self) -> None:
        """
        pointsconfig: dictionary with points per score type as per dream11
        :return:
        """
        self.player_scorecard['total_wickets_points'] = self.pointsconfig['total_wickets'] * self.player_scorecard['total_wickets']
        self.player_scorecard['economy_rate_points'] = np.where(self.player_scorecard['economy_rate'] >= 11, self.pointsconfig['>11E'],
                                                           np.where(self.player_scorecard['economy_rate'] >= 10, self.pointsconfig['>10E'],
                                                                    np.where(self.player_scorecard['economy_rate'] >= 9,self.pointsconfig['>9E'],
                                                                             np.where(self.player_scorecard['economy_rate'] <= 4,self.pointsconfig['<=4E'],
                                                                                 np.where(self.player_scorecard['economy_rate'] <= 5,self.pointsconfig['<5E'],
                                                                                          np.where(self.player_scorecard['economy_rate'] <= 6, self.pointsconfig['<6E'],0))))))
        self.player_scorecard['maiden_overs_points'] = self.pointsconfig['maiden_overs'] * self.player_scorecard['maiden_overs']
        self.player_scorecard['wicket_bonus_points'] = np.where(self.player_scorecard['total_wickets'] >= 5, self.pointsconfig['>=5W'],
                                                           np.where(self.player_scorecard['total_wickets'] >= 4, self.pointsconfig['>=4W'], 0))
        self.player_scorecard['total_bowl_points'] = np.nan
        self.player_scorecard['total_bowl_points'] = self.player_scorecard['total_wickets_points'].\
                                                add(self.player_scorecard['maiden_overs_points'], fill_value=0). \
                                                add(self.player_scorecard['wicket_bonus_points'], fill_value=0).\
                                                add(self.player_scorecard['economy_rate_points'], fill_value=0)
        self.player_scorecard['total_bowl_points'] = np.where(self.player_scorecard['total_balls_bowled'] >= 1, self.player_scorecard['total_bowl_points'], np.nan)
        return

    def get_batsmen_bowler_points(self):
        self.get_batting_points()
        self.get_bowling_points()
        self.player_scorecard['total_points'] = self.player_scorecard['total_bat_points'].add(self.player_scorecard['total_bowl_points'], fill_value=0)
        return


class FeatEngineering:

    def __init__(self, ipl_features, matchsummary):
        self.ipl_features = ipl_features
        matchsummary['city'] = np.where(matchsummary['city'].isin(['Bangalore', 'Bengaluru']), 'Bengaluru', matchsummary['city'])
        matchsummary['venue'] = np.where(matchsummary['venue'].isin(['M Chinnaswamy Stadium', 'M.Chinnaswamy Stadium']),'M Chinnaswamy Stadium',matchsummary['venue'])
        matchsummary['venue'] = np.where(matchsummary['venue'].isin(['Punjab Cricket Association IS Bindra Stadium, Mohali', 'Punjab Cricket Association Stadium, Mohali']),'Punjab Cricket Association Stadium', matchsummary['venue'])
        self.matchsummary = matchsummary
        if "playing_team" not in self.ipl_features.columns:
            self.ipl_features['playing_team'] = np.where(pd.isnull(self.ipl_features['batsmen_innings']), self.ipl_features['bowler_bowlingteam'], self.ipl_features['batsmen_battingteam'])
        if "opposition_team" not in ipl_features.columns:
            self.ipl_features['opposition_team'] = np.where(pd.isnull(self.ipl_features['batsmen_innings']), self.ipl_features['bowler_battingteam'], self.ipl_features['batsmen_bowlingteam'])
            self.ipl_features[['playing_team', 'opposition_team']].replace({'Delhi Daredevils': 'Delhi Capitals', 'Rising Pune Supergiants': 'Pune Warriors',
                                                                        'Rising Pune Supergiant': 'Pune Warriors', 'Deccan Chargers': 'Sunrisers Hyderabad'}, inplace=True)

        return

    def add_venue_info(self):
        """

        :param matchsummary:
        :return:
        """

        self.ipl_features = pd.merge(self.ipl_features, self.matchsummary[['matchid', 'year', 'city', 'venue','team1','team2','toss_winner']], on='matchid', how='left')

        self.ipl_features.rename(columns={'team1': "home_team", 'team2': 'away_team'}, inplace=True)

        # # add a feature to check if its the home game for the player
        # self.add_homegame_flag()
        # # add a feature to check if the player's team won the toss
        # self.add_toss_info()
        return
    def add_homegame_flag(self):
        self.ipl_features['home_game'] = np.where((self.ipl_features['playing_team'] == self.ipl_features['home_team']),1, 0)
        return

    def add_toss_info(self):
        self.ipl_features['toss_flag'] = np.where((self.ipl_features['playing_team'] == self.ipl_features['toss_winner']), 1, 0)
        return

    def add_player_match_count(self):
        self.ipl_features = self.ipl_features.sort_values(by=['matchid', 'playername'], ascending=True)
        self.ipl_features['player_match_count'] = 1
        self.ipl_features['player_match_count'] = self.ipl_features.groupby(['playername'])['player_match_count'].cumsum()
        return


    def add_lagging_feat(self, match_id, groupby_id, rolling_window, *args):
        """
        add rolling average feature to add the average batting and bowling points scored by the player, strike rate, economy rate, average batting position
        :return:
        """
        ipl_features = self.ipl_features.sort_values(by=[match_id, groupby_id], ascending=True)
        for col in args:
            print('col:', col)
            outcolname = col + "_" + groupby_id + '_avg' + str(rolling_window)
            # why are we removing duplicates ?
            rolling_avg_points = ipl_features[[match_id, groupby_id, col]].drop_duplicates()
            #
            rolling_avg_points = pd.DataFrame(rolling_avg_points.groupby([match_id, groupby_id])[col].sum()).reset_index()
            rolling_avg_points.set_index(match_id, inplace=True)
            rolling_avg_points = pd.DataFrame(rolling_avg_points.groupby([groupby_id])[col].rolling(rolling_window).mean()).reset_index().rename(columns={col: outcolname})
            rolling_avg_points[outcolname] = pd.DataFrame(rolling_avg_points.groupby([groupby_id])[outcolname].shift(1))
            self.ipl_features = pd.merge(self.ipl_features, rolling_avg_points, on=[match_id, groupby_id], how='left')
        return


    def add_player_leanpatch(self):
        self.ipl_features['lean_patch_3'] = np.where( np.isnan(self.ipl_features['totalpoints_playername_avg_10']),0,
                                                      np.where(self.ipl_features['totalpoints_playername_avg_3'] < self.ipl_features['totalpoints_playername_avg_10']*.6, 1, 0))
        self.ipl_features['lean_patch_2'] = np.where( np.isnan(self.ipl_features['totalpoints_playername_avg_5']),0,
                                                      np.where(self.ipl_features['totalpoints_playername_avg_2'] < self.ipl_features['totalpoints_playername_avg_5']*.6, 1, 0))
        self.ipl_features['lean_patch_5'] = np.where( np.isnan(self.ipl_features['totalpoints_playername_avg_10']),0,
                                                      np.where(self.ipl_features['totalpoints_playername_avg_5'] < self.ipl_features['totalpoints_playername_avg_10']*.6, 1, 0))

