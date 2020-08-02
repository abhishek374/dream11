import pandas as pd
import numpy as np

class ScoreCard:

    def __init__(self, matchdata):
        self.matchdata = matchdata
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
        bowler_wickets = pd.DataFrame(self.matchdata[(self.matchdata['dismissal'] == 't') & (~self.matchdata['dismissedtype'].isin(['run out','retired hurt']))].
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
        df_list = [bowler_wickets, bowler_overs_bow, bowler_ball_faced_legal, bowler_runs_given,matchdata_runs_per_over]
        bowler_summary = pd.concat(df_list, join='outer', axis=1).fillna(np.nan).reset_index()
        # adding additional columns just in case we need for modeling
        bowler_summary = pd.merge(bowler_summary, self.matchdata[['matchid', 'bowlername', 'innings', 'battingteam', 'bowlingteam']].drop_duplicates(), on=['matchid', 'bowlername'], how='left')
        bowler_summary.rename(columns={'innings': 'bowlers_innings', 'bowlername': 'playername', 'battingteam': 'bowler_battingteam',
                                       'bowlingteam': 'bowler_bowlingteam'}, inplace=True)
        bowler_summary['economy_rate'] = (bowler_summary['total_runs_given'] * 6) / bowler_summary['total_legal_balls_bowled']
        self.bowler_summary = bowler_summary
        return

    def merge_player_scorecard(self) -> pd.DataFrame:
        """
        batting_points: scorecard from batsmen perspective with dream11 points
        bowling_points: scorecard from bowling perspective with dream11 points
        :return: ipl_points: merged dataset with the total points per player in a match
        """
        self.batsmen_summary_fun()
        self.bowler_summary_fun()
        ipl_points = pd.merge(self.batsmen_summary, self.bowler_summary, on=['matchid', 'playername'], how='outer')
        player_avg = self.get_player_role(ipl_points)
        ipl_points = pd.merge(ipl_points, player_avg[['playername', 'playing_role']], on='playername', how='left')
        return ipl_points

    def get_player_role(self, input_df) -> pd.DataFrame:
        """"
        function to get the players role in the team
        input_df: input dataframe that has the combined scorecard of the players with batting and bowling
        :return player_avg: df with the player name and player's playing role
        """
        MINAVGBALLSFACED = 8
        MINAVGBOWLSBOWLED = 6
        player_avg = input_df[['playername', 'total_balls_faced', 'total_balls_bowled']].fillna(0)
        player_avg = pd.DataFrame(player_avg.groupby('playername')['total_balls_faced', 'total_balls_bowled'].mean())
        conditions = [((player_avg['total_balls_faced'] >= MINAVGBALLSFACED) & (player_avg['total_balls_bowled'] >= MINAVGBOWLSBOWLED)),
                      (player_avg['total_balls_bowled'] >= MINAVGBOWLSBOWLED)]
        choices = ['All Rounder', 'Bowler']
        player_avg['playing_role'] = np.select(conditions, choices, default='Batsmen')
        player_avg = player_avg.reset_index()
        return player_avg


class Dream11Points:

    def __init__(self, player_scorecard,pointsconfig):
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
        self.player_scorecard['total_bat_points'] = 0
        self.player_scorecard['total_bat_points'] = self.player_scorecard['total_runs_points'].add(self.player_scorecard['run_6_points'], fill_value=0). \
            add(self.player_scorecard['run_4_points'], fill_value=0).add(self.player_scorecard['run_bonus_points'], fill_value=0)
        return

    def get_bowling_points(self) -> None:
        """
        pointsconfig: dictionary with points per score type as per dream11
        :return:
        """
        self.player_scorecard['total_wickets_points'] = self.pointsconfig['total_wickets'] * self.player_scorecard['total_wickets']
        self.player_scorecard['economy_rate_points'] = np.where(self.player_scorecard['economy_rate'] >= 11, self.pointsconfig['>11E'],
                                                           np.where(self.player_scorecard['economy_rate'] >= 10,self.pointsconfig['>10E'],
                                                                    np.where(self.player_scorecard['economy_rate'] >= 9,self.pointsconfig['>9E'],
                                                                             np.where(self.player_scorecard['economy_rate'] <= 4,self.pointsconfig['<=4E'],
                                                                                 np.where(self.player_scorecard['economy_rate'] <= 5,self.pointsconfig['<5E'],
                                                                                          np.where(self.player_scorecard['economy_rate'] <= 6, self.pointsconfig['<6E'],0))))))
        self.player_scorecard['maiden_overs_points'] = self.pointsconfig['maiden_overs'] * self.player_scorecard['maiden_overs']
        self.player_scorecard['wicket_bonus_points'] = np.where(self.player_scorecard['total_wickets'] >= 5, self.pointsconfig['>=5W'],
                                                           np.where(self.player_scorecard['total_wickets'] >= 4, self.pointsconfig['>=4W'], 0))
        self.player_scorecard['total_bowl_points'] = 0
        self.player_scorecard['total_bowl_points'] = self.player_scorecard['total_wickets_points'].\
                                                add(self.player_scorecard['maiden_overs_points'], fill_value=0). \
                                                add(self.player_scorecard['wicket_bonus_points'], fill_value=0).\
                                                add(self.player_scorecard['economy_rate_points'], fill_value=0)
        return

    def get_batsmen_bowler_points(self) -> pd.DataFrame:
        self.get_batting_points()
        self.get_bowling_points()
        self.player_scorecard['total_points'] = self.player_scorecard['total_bat_points'].add(self.player_scorecard['total_bowl_points'], fill_value=0)
        return self.player_scorecard
