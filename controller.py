import pandas as pd
from main import *

if __name__ == "__main__":
    # reading the source file from local
    matchdatapath = r'Data/matchdata.csv'
    matchsummarypath = r'Data/matchsummary.csv'
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
        '1per': 5000,  # 10000
        '2per': 3000,  # 6000
        '3per': 500,  # 500
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
                 'VENUE': 'venue',
                 'TOTALBALLSBOWLED':'total_balls_bowled',
                 'BATTINGORDER': 'fallofwickets',
                 'BATTINGTEAM': 'battingteam',
                 'BOWLINGTEAM': 'bowlingteam',
                 'PLAYERNAME': 'playername',
                 'TOTALBATPOINTS': 'total_bat_points',
                 'TOTALBALLPOINTS': 'total_bowl_points',
                 'ACTUALPOINTS': 'total_points',
                 'PREDPOINTS': 'pred_points',
                 'PLAYERCOST': 'playercost',
                 'PLAYINGROLE': 'playing_role',
                 'PREDSELECTION': 'pred_selection_true',
                 'ACTUALSELECTION': 'actual_selection_true',
                 'PREDSELECTIONRANK': 'pred_selection_rank',
                 'ACTUALSELECTIONRANK': 'actual_selection_rank'}

    predictors = ['playing_team', 'opposition_team', 'playing_role', 'city', 'home_game', 'toss_flag','player_match_count',
                  'fallofwickets_playername_avg2', 'total_balls_bowled_playername_avg2',
                  'total_bat_points_venue_avg2', 'total_bowl_points_venue_avg2', 'total_points_playername_avg2',
                  'total_points_playername_avg5', 'fallofwickets_playername_avg5', 'total_balls_bowled_playername_avg5',
                  'total_bat_points_venue_avg5', 'total_bowl_points_venue_avg5',
                  'fallofwickets_playername_avg10', 'total_balls_bowled_playername_avg10',
                  'total_bat_points_venue_avg10', 'total_bowl_points_venue_avg10', 'total_points_playername_avg10',
]

    cat_cols = ['playing_team', 'playing_role', 'opposition_team', 'city']
    target_col = 'total_points'
    pred_col = 'pred_points'

    modelname = 'xgb'
    matchdatascorecardpath = r'Data/ipl_scorecard_points.csv'
    featenggpath = r'Data/ipl_scorecard_points_featengg.csv'

    modelpath = r"Data/" + modelname + "_model.pkl"
    encoderpath = r"Data/OnHotEncoder_" + modelname + ".pkl"

    modelresultspath = r"Data/model_prediction.csv"
    predfeaturepath = r"Data/pred_data_features.csv"
    predscorecardpath = r"Data/pred_data_scorecard.csv"
    predsummarypath = r"Data/pred_data_summary.csv"
    nextmatchteampath = r"Data/pred_team11.csv"

    datapath = {'matchdatapath': matchdatapath,
                'matchsummarypath': matchsummarypath,
                'matchdatascorecardpath': matchdatascorecardpath,
                'featenggpath': featenggpath,
                'modelpath': modelpath,
                'encoderpath': encoderpath,
                'modelresultspath': modelresultspath,
                'predscorecardpath': predscorecardpath,
                'predsummarypath': predsummarypath,
                'predfeaturepath': predfeaturepath,
                'nextmatchteampath': nextmatchteampath}

    model_train = False
    full_model_predict = False
    next_match_team = False

    # Run the below function to train the model
    if model_train:
        execute_get_scorecard(datapath, pointsconfig)  # Run the function to to get points in the scorecard format
        execute_featureengg(datapath['matchdatascorecardpath'], datapath['matchsummarypath'], datapath['featenggpath'], colconfig)  # Run the function to create features
        execute_model_train(datapath, modelname, predictors, cat_cols, target_col, usetimeseries=False)  # Run the function to build the model

    # Run the below function to predict using the saved model on the complete dataset
    if full_model_predict:
        execute_model_prediction(datapath, predictors, modelname, cat_cols, pred_col, usetimeseries=False)  # Run the function to predict the points based on the model
        execute_team_selection(datapath, constconfig, colconfig)  # Run the function to only select the predicted playing 11
        execute_rewards_calcualtion(datapath, constconfig, colconfig, rewardconfig)  # Run the function to estimate rewards if actual playing 11 is available

    # Run the below function to predict the best 11 for the upcoming match
    if next_match_team:
        finalteam = pd.DataFrame()
        # Change the values of team1, team2, city and venue depending on the match
        TEAM1 = "Kolkata Knight Riders"
        TEAM2 = "Royal Challengers Bangalore"
        CITY = 'Abu Dhabi'
        VENUE = 'Sheikh Zayed Stadium'
        create_pred_dataframe(datapath, colconfig, TEAM1, TEAM2,CITY, VENUE, toss_winner=TEAM1)
        for modelname in ['catboost', 'xgb', 'movingaverage']:
            modelpath = r"Data/" + modelname + "_model.pkl"
            encoderpath = r"Data/OnHotEncoder_" + modelname + ".pkl"
            datapath['modelpath'] = modelpath
            datapath['encoderpath'] = encoderpath
            execute_model_prediction(datapath, predictors, modelname, cat_cols, pred_col, usetimeseries=False, predpath=True)
            teamtemp = execute_team_selection(datapath, constconfig, colconfig).team_points
            teamtemp.sort_values(by=['pred_selection_true', 'pred_points', 'playername'], inplace=True, ascending=False)
            teamtemp.rename(columns={'pred_points': 'pred_points' + '_' + modelname, 'pred_selection_true': 'pred_selection_true' + '_' + modelname}, inplace=True)
            teamtemp = teamtemp[['playername', 'playing_role', 'playing_team', 'pred_points' + '_' + modelname,'pred_selection_true' + '_' + modelname]]
            if finalteam.shape[0] == 0:
                finalteam = teamtemp
            else:
                finalteam = pd.merge(finalteam, teamtemp, on=['playername', 'playing_role', 'playing_team'],how='left')
        finalteam.to_csv(nextmatchteampath, index=False)

        #TODO make the constraint for allrounder, batsmen, bowler dynamic
        #TODO option to add weights manually
