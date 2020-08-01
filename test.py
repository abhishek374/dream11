import pandas as pd
import numpy as np

# reading the source file from local
matchdata = pd.read_csv(r'matchdata.csv')
class ScoreCard:
    def __init__(self, matchdata):

        self.matchdata = matchdata
        self.matchdata['over_num'] = self.matchdata['over'].apply(lambda x: int(x))
        return

    def batsmen_summary_fun(self) -> pd.DataFrame:
        """
        matchdata: pandas dataframe of ball by ball data for the matches

        return: batsmen_summary: pandas datarame with summary scorecard of a batsmen
        """
        batsmen_score = pd.DataFrame(self.matchdata.groupby(['matchid', 'batsmanname'])['scorevalue'].sum()).\
            rename(columns={"scorevalue": "total_runs"})
        batsmen_ball_faced = pd.DataFrame(self.matchdata.groupby(['matchid', 'batsmanname'])['over'].count()).\
            rename(columns={"over": "total_balls"})
        batsmen_ball_faced_legal = pd.DataFrame(self.matchdata.groupby(['matchid', 'batsmanname'])['over'].nunique()).\
            rename(columns={"over": "total_legal_balls"})
        batsmen_scores6 = pd.DataFrame(self.matchdata[self.matchdata['scorevalue'] == 6].groupby(['matchid', 'batsmanname'])['scorevalue'].count()).\
            rename(columns={"scorevalue": "run_6"})
        batsmen_scores4 = pd.DataFrame(self.matchdata[self.matchdata['scorevalue'] == 4].groupby(['matchid', 'batsmanname'])['scorevalue'].count()).\
            rename(columns={"scorevalue": "run_4"})
        df_list = [batsmen_score, batsmen_ball_faced, batsmen_ball_faced_legal, batsmen_scores6, batsmen_scores4]
        batsmen_summary = pd.concat(df_list, join='outer', axis=1).fillna(np.nan).reset_index()
        batsmen_summary = pd.merge(batsmen_summary, self.matchdata[['matchid', 'batsmanname', 'innings', 'battingteam', 'bowlingteam']].
                                   drop_duplicates(), on=['matchid', 'batsmanname'], how='left')
        batsmen_summary.rename(columns={'innings': 'batsmen_innings'}, inplace=True)
        return batsmen_summary

    def bowler_summary_fun(self) -> pd.DataFrame:
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
        bowler_summary.rename(columns={'innings': 'bowlers_innings'}, inplace=True)
        bowler_summary['economy_rate'] = (bowler_summary['total_runs_given'] * 6)/bowler_summary['total_legal_balls_bowled']
        return bowler_summary

# getting the scorecard from a batsmen's perspective
ipl_scorecard  = ScoreCard(matchdata.copy())
batsmen_scorecard = ipl_scorecard.batsmen_summary_fun()
# getting the scorecard from a bowler's perspective
bowler_scorecard = ipl_scorecard.bowler_summary_fun()

# defining the points system
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


def get_batting_points(batsmen_scorecard, pointsconfig) -> pd.DataFrame:
    """
    pointsconfig: dictionary with points per score type as per dream11
    :return:
    """
    batsmen_scorecard['total_runs_points'] = pointsconfig['total_runs'] * batsmen_scorecard['total_runs']
    batsmen_scorecard['run_6_points'] = pointsconfig['run_6'] * batsmen_scorecard['run_6']
    batsmen_scorecard['run_4_points'] = pointsconfig['run_4'] * batsmen_scorecard['run_4']
    batsmen_scorecard['run_bonus_points'] = np.where(batsmen_scorecard['total_runs'] == 0, pointsconfig['duck'], 0)
    batsmen_scorecard['run_bonus_points'] = np.where(batsmen_scorecard['total_runs'] >= 50, pointsconfig['>=50'] +
                                                     batsmen_scorecard['run_bonus_points'], batsmen_scorecard['run_bonus_points'])
    batsmen_scorecard['run_bonus_points'] = np.where(batsmen_scorecard['total_runs'] >= 100, pointsconfig['>=100'] +
                                                     batsmen_scorecard['run_bonus_points'], batsmen_scorecard['run_bonus_points'])
    batsmen_scorecard['total_bat_points'] = 0
    batsmen_scorecard['total_bat_points'] = batsmen_scorecard['total_bat_points'].add(batsmen_scorecard['run_6_points'], fill_value=0).\
        add(batsmen_scorecard['run_4_points'], fill_value=0).add(batsmen_scorecard['run_bonus_points'], fill_value=0)
    return batsmen_scorecard

def get_bowling_points(bowler_scorecard,pointsconfig) -> pd.DataFrame:
    """
    pointsconfig: dictionary with points per score type as per dream11
    :return:
    """
    bowler_scorecard['total_wickets_points'] = pointsconfig['total_wickets'] * bowler_scorecard['total_wickets']
    bowler_scorecard['economy_rate_points'] = np.where(bowler_scorecard['economy_rate'] >= 11, pointsconfig['>11E'],
                                                       np.where(bowler_scorecard['economy_rate'] >= 10, pointsconfig['>10E'],
                                                                np.where(bowler_scorecard['economy_rate'] >= 9, pointsconfig['>9E'],
                                                                         np.where(bowler_scorecard['economy_rate'] <= 4, pointsconfig['<=4E'],
                                                                                  np.where(bowler_scorecard['economy_rate'] <= 5, pointsconfig['<5E'],
                                                                                           np.where(bowler_scorecard['economy_rate'] <= 6, pointsconfig['<6E'], 0))))))
    bowler_scorecard['maiden_overs_points'] = pointsconfig['maiden_overs'] * bowler_scorecard['maiden_overs']
    bowler_scorecard['wicket_bonus_points'] = np.where(bowler_scorecard['total_wickets'] >= 5, pointsconfig['>=5W'],
                                                       np.where(bowler_scorecard['total_wickets'] >= 4, pointsconfig['>=4W'],0))
    bowler_scorecard['total_bowl_points'] = 0
    bowler_scorecard['total_bowl_points'] = bowler_scorecard['total_bowl_points'].add(bowler_scorecard['maiden_overs_points'], fill_value=0).\
        add(bowler_scorecard['wicket_bonus_points'], fill_value=0).add(bowler_scorecard['economy_rate_points'], fill_value = 0)
    return bowler_scorecard


batting_points = get_batting_points(batsmen_scorecard, pointsconfig)
print(batting_points)

bowling_points = get_bowling_points(bowler_scorecard, pointsconfig)
print(bowling_points)

