import pandas as pd
from main import *
from send_email import send_email_team
from point_prediction import EnsembleModel

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

    constconfig = {'MAXCOSTPOINT': 100,
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
                 'TOTALBALLSBOWLED': 'total_balls_bowled',
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

    modelname = 'catboost'  # Options include 'rf','xgb','catboost','movingaverage', 'ensemble'
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

    # Change the below to true to run the training of the models part of the permissible list
    TRAIN_MODEL = False
    # Change the below to true to run the prediction on the entire training dataset we have
    PREDICT_MODEL = False
    # Change the below to true to run the training for an ensemble model using predicitons from other model
    PREDICT_ENSEMBLE = False
    # Change the below to true to create the dataframe of the upcoming match and adjust anything if required
    SELECT_PLAYING_SQUAD = False
    # Change the below to true if the squad file is ready at predfeaturepath to run prediction for the team
    SELECT_CURRENT_TEAM = False
    # Change this to true to send email if the file fo next match is present at nextmatchteampath
    SEND_EMAIL = True
    modelnamelist = ['xgb', 'catboost', 'rf', 'movingaverage']
    #modelnamelist = ['catboost']
    # Run the below function to train the model
    if TRAIN_MODEL:
        execute_get_scorecard(datapath, pointsconfig)  # Run the function to to get points in the scorecard format
        execute_featureengg(datapath['matchdatascorecardpath'], datapath['matchsummarypath'], datapath['featenggpath'], colconfig)  # Run the function to create features
        execute_model_train(datapath, modelname, predictors, cat_cols, target_col, usetimeseries=False)  # Run the function to build the model

    # Run the below function to predict using the saved model on the complete dataset
    if PREDICT_MODEL:
        finalmodelpred = pd.read_csv(datapath['featenggpath'])
        finalmodelpred = finalmodelpred[['matchid', 'playername', 'playing_role', 'playing_team', 'playercost', 'total_points']]
        predcol_list = []
        for modelname in modelnamelist:
            modelpath = r"Data/" + modelname + "_model.pkl"
            encoderpath = r"Data/OnHotEncoder_" + modelname + ".pkl"
            datapath['modelpath'] = modelpath
            datapath['encoderpath'] = encoderpath
            modelpred_df = execute_model_prediction(datapath, predictors, modelname, cat_cols, pred_col + '_' + modelname, usetimeseries=False)  # Run the function to predict the points based on the model
            modelpred_df = modelpred_df[['matchid','playername', 'playing_role', 'playing_team', 'playercost', 'total_points',pred_col + '_' + modelname]]
            predcol_list.append(pred_col + '_' + modelname)
            print(f"prediction for model {modelname} is complete")
            finalmodelpred = pd.merge(finalmodelpred, modelpred_df, on=['matchid','playername', 'playing_role', 'playing_team', 'playercost', 'total_points'], how='left')

        if PREDICT_ENSEMBLE:
            EM = EnsembleModel()
            temp = EM.get_ensemble_model_train(finalmodelpred, predcol_list, target_col='total_points',predcol= pred_col, modelpath=r"Data/" + 'ensemble' + "_model.pkl")
            finalmodelpred = pd.merge(finalmodelpred, temp, left_index=True, right_index=True, how='left')
        else:
            finalmodelpred[pred_col] = finalmodelpred[pred_col + '_' + modelname]
        finalmodelpred.to_csv(datapath['modelresultspath'], index=False)
        execute_team_selection(datapath, constconfig, colconfig)  # Run the function to only select the predicted playing 11
        execute_rewards_calcualtion(datapath, constconfig, colconfig, rewardconfig)  # Run the function to estimate rewards if actual playing 11 is available

    # Enter the details of the current match/
    TEAM1 = "Delhi Capitals"
    TEAM2 = "Kings XI Punjab"
    CITY = 'neutral venue'
    VENUE = 'Dubai International Cricket Stadium'
    # Run the below function to predict the best 11 for the upcoming match

    if SELECT_PLAYING_SQUAD:
        # Change the values of team1, team2, city and venue depending on the match
        create_pred_dataframe(datapath, colconfig, TEAM1, TEAM2, CITY, VENUE, toss_winner=TEAM1)
    if SELECT_CURRENT_TEAM:
        finalteam = pd.DataFrame()
        predcol_list = []
        for modelname in ['catboost', 'xgb', 'rf', 'movingaverage', 'ensemble']: # Keep ensemble in the end
            modelpath = r"Data/" + modelname + "_model.pkl"
            encoderpath = r"Data/OnHotEncoder_" + modelname + ".pkl"
            datapath['modelpath'] = modelpath
            datapath['encoderpath'] = encoderpath
            if modelname == 'ensemble':
                EnsembleModel().get_ensemble_model_pred(datapath, finalteam, predcol_list, pred_col)
            else:
                execute_model_prediction(datapath, predictors, modelname, cat_cols, pred_col, usetimeseries=False, predpath=True)
            teamtemp = execute_team_selection(datapath, constconfig, colconfig).team_points
            teamtemp.sort_values(by=['matchid','pred_selection_true', 'pred_points', 'playername'], inplace=True, ascending=False)
            teamtemp.rename(columns={'pred_points': 'pred_points' + '_' + modelname, 'pred_selection_true': 'pred_selection_true' + '_' + modelname}, inplace=True)
            teamtemp = teamtemp[['matchid','playername', 'playing_role', 'playing_team', 'playercost','pred_points' + '_' + modelname, 'pred_selection_true' + '_' + modelname]]
            predcol_list.append('pred_points' + '_' + modelname)
            if finalteam.shape[0] == 0:
                finalteam = teamtemp
            else:
                finalteam = pd.merge(finalteam, teamtemp, on=['matchid','playername', 'playing_role', 'playing_team','playercost'], how='left')
        # collist = [col for col in finalteam.columns if 'pred_selection' in col]
        # finalteam['pred_selection_all'] = finalteam[collist].apply(np.sum, axis=1)
        # finalteam['pred_selection_all'] = np.where(finalteam['pred_selection_all'] == len(collist), 1, "")
        finalteam = formatdata(finalteam)
        finalteam.to_csv(nextmatchteampath, index=False)
    if SEND_EMAIL:
        send_email_team(TEAM1, TEAM2, nextmatchteampath)
