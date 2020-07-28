import pandas as pd
import numpy as np

# reading the source file from local
matchdata = pd.read_csv(r'matchdata.csv')

def batsmen_summary_fun(matchdata) -> pd.DataFrame :
    """
    matchdata: pandas dataframe of ball by ball data for the matches

    return: batsmen_summary: pandas datarame with summary scorecard of a batsmen
    """
    batsmen_score = pd.DataFrame(matchdata.groupby(['matchid', 'batsmanname'])['scorevalue'].sum()).\
        rename(columns={"scorevalue":"total_runs"})
    batsmen_ball_faced = pd.DataFrame(matchdata.groupby(['matchid', 'batsmanname'])['over'].count()).\
        rename(columns={"over":"total_balls"})
    batsmen_ball_faced_legal = pd.DataFrame(matchdata.groupby(['matchid', 'batsmanname'])['over'].nunique()).\
        rename(columns={"over":"total_legal_balls"})
    batsmen_scores6 = pd.DataFrame(matchdata[matchdata['scorevalue'] == 6].groupby(['matchid', 'batsmanname'])['scorevalue'].count()).\
        rename(columns={"scorevalue": "run_6"})
    batsmen_scores4 = pd.DataFrame(matchdata[matchdata['scorevalue'] == 4].groupby(['matchid', 'batsmanname'])['scorevalue'].count()).\
        rename(columns={"scorevalue": "run_4"})
    df_list = [batsmen_score, batsmen_ball_faced, batsmen_ball_faced_legal, batsmen_scores6,batsmen_scores4]
    batsmen_summary = pd.concat(df_list, join='outer', axis=1).fillna(np.nan).reset_index()
    batsmen_summary = pd.merge(batsmen_summary, matchdata[['matchid', 'batsmanname', 'innings', 'battingteam', 'bowlingteam']].
                               drop_duplicates(), on=['matchid', 'batsmanname'], how='left')
    batsmen_summary.rename(columns={'innings': 'batsmen_innings'}, inplace=True)
    return batsmen_summary

def bowler_summary_fun(matchdata) -> pd.DataFrame :
    """
    matchdata: pandas dataframe of ball by ball data for the matches

    return: bowler_summary: pandas datarame with summary scorecard of a bowler
    """
    bowler_wickets = pd.DataFrame(matchdata[(matchdata['dismissal'] == 't') & (~matchdata['dismissedtype'].isin(['run out','retired hurt']))].
                                  groupby(['matchid', 'bowlername'])['dismissal'].count()).\
        rename(columns={"dismissal": "total_wickets"})
    bowler_overs_bow = pd.DataFrame(matchdata.groupby(['matchid', 'bowlername'])['over'].count()).\
        rename(columns={"over": "total_balls_bowled"})
    bowler_ball_faced_legal = pd.DataFrame(matchdata.groupby(['matchid', 'bowlername'])['over'].nunique()).\
        rename(columns={"over": "total_legal_balls_bowled"})
    # calculating the number of maiden overs bowled in a match
    bowler_runs_given = pd.DataFrame(matchdata.groupby(['matchid', 'bowlername'])['batsmanscorevalue'].sum()).\
        rename(columns={"batsmanscorevalue": "total_runs_given"})
    matchdata['over_num'] = matchdata['over'].apply(lambda x: int(x))
    matchdata_runs_per_over = pd.DataFrame(matchdata.groupby(['matchid', 'bowlername', 'over_num'])['batsmanscorevalue'].sum()).reset_index()
    matchdata_runs_per_over['maiden_overs'] = np.where(matchdata_runs_per_over['batsmanscorevalue'] == 0, 1, 0)
    matchdata_runs_per_over = matchdata_runs_per_over.groupby(['matchid', 'bowlername'])['maiden_overs'].sum()
    # combining all the features created so far
    df_list = [bowler_wickets, bowler_overs_bow, bowler_ball_faced_legal, bowler_runs_given,matchdata_runs_per_over]
    bowler_summary = pd.concat(df_list, join='outer', axis=1).fillna(np.nan).reset_index()
    # adding additional columns just in case we need for modeling
    bowler_summary = pd.merge(bowler_summary, matchdata[['matchid', 'bowlername', 'innings', 'battingteam', 'bowlingteam']].drop_duplicates(), on=['matchid', 'bowlername'], how='left')
    bowler_summary.rename(columns={'innings': 'bowlers_innings'}, inplace=True)
    return bowler_summary

# getting the scorecard from a batsmen's perspective
batsmen_scorecard = batsmen_summary_fun(matchdata.copy())
# getting the scorecard from a bowler's perspective
bowler_scorecard = bowler_summary_fun(matchdata.copy())
