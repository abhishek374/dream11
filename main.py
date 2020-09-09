from data_prep import ScoreCard, Dream11Points, FeatEngineering
from optimized_selection import SelectPlayingTeam,RewardEstimate
import pandas as pd
import numpy as np
import pickle
from point_prediction import ModelTrain, ModelPredict

##################Part to create additional features for modeling#######################################################

def execute_get_scorecard(datapath,pointsconfig):
    """

    :return:
    """
    # getting the scorecard from a batsmen's perspective
    matchdata = pd.read_csv(datapath['matchdatapath'])
    ipl_scorecard = ScoreCard(matchdata)
    # merging both the batsmen and bowler's points to get a single view
    ipl_scorecard.merge_player_scorecard()
    # calculating the points scored by the players based on dream11 scoring method
    ipl_scorecard_points = Dream11Points(ipl_scorecard.ipl_merged_scorecard, pointsconfig)
    ipl_scorecard_points.get_batsmen_bowler_points()
    ipl_scorecard_points.player_scorecard.to_csv(datapath['matchdatascorecardpath'])
    return

##################Part to create additional features for modeling###################################################
def execute_featureengg(matchdatascorecardpath,matchsummarypath, featenggpath,colconfig):
    """

    :param matchdatascorecardpath:
    :param matchsummary:
    :param featenggpath:
    :return:
    """
    points_df = pd.read_csv(matchdatascorecardpath)
    matchsummary = pd.read_csv(matchsummarypath)
    FeatEng = FeatEngineering(points_df, matchsummary.copy())
    FeatEng.add_venue_info()
    FeatEng.add_homegame_flag()
    FeatEng.add_toss_info()
    FeatEng.add_player_match_count()

    rolling_window = [2, 3, 5, 10]
    for i in rolling_window:
        FeatEng.add_lagging_feat(colconfig['MATCHID'], colconfig['VENUE'], i, colconfig['TOTALBATPOINTS'], colconfig['TOTALBALLPOINTS'])
        FeatEng.add_lagging_feat(colconfig['MATCHID'], colconfig['PLAYERNAME'], i, colconfig['ACTUALPOINTS'],  colconfig['BATTINGORDER'], colconfig['TOTALBALLSBOWLED'])
    # writing the scorecard to save it
    # Temp till we get better alternate to cost of each player
    FeatEng.ipl_features['playercost'] = 10
    FeatEng.ipl_features.to_csv(featenggpath, index=False)
    return FeatEng.ipl_features


##################Part to train the model #################################################

def execute_model_train(datapath,modelname, predictors, cat_cols, target_col, usetimeseries=False):
    """

    :return:
    """
    masterdf = pd.read_csv(datapath['featenggpath'])
    if usetimeseries:
        ts_prediction = ModelTrain.get_timeseries_forecast(masterdf, target_col, 'playername', 'ts_pred_points')
        masterdf = pd.merge(masterdf, ts_prediction, left_index=True, right_index=True, how='left')
        masterdf.to_csv(r'Data\time_series_output.csv', index=False)
        predictors = predictors + ['ts_pred_points']
    if modelname == 'movingaverate':
        return
    modeltrain = ModelTrain(masterdf, target_col, predictors, cat_cols, modelname)
    modeltrain.get_test_train(split_col='year', split_value=[2019])
    modeltrain.get_normalized_data()
    modelobjects = modeltrain.train_model(model=modelname)
    pickle.dump(modelobjects[2], open(datapath['modelpath'], 'wb'))
    pickle.dump(modelobjects[:2], open(datapath['encoderpath'], 'wb'))
    print(modeltrain.feat_imp_df)
    return


def execute_model_prediction(datapath,  predictors, modelname, cat_cols, pred_col, usetimeseries=False, predpath=False):
    """

    :return:
    """

    if predpath:
        masterdf = pd.read_csv(datapath['predfeaturepath'])
    else:
        masterdf = pd.read_csv(datapath['featenggpath'])

    if modelname == 'rf':
        masterdf.fillna(-100, inplace=True)

    if usetimeseries:
        predictors = predictors + ['ts_pred_points']

    if modelname == 'movingaverage':
        masterdf[pred_col] = masterdf['total_points_playername_avg3']
        masterdf.to_csv(datapath['modelresultspath'], index=False)
        return

    modelpkl = pickle.load(open(datapath['modelpath'], 'rb'))
    enc = pickle.load(open(datapath['encoderpath'], 'rb'))
    mod_predict = ModelPredict(masterdf, enc, modelpkl, modelname, predictors, cat_cols, pred_col)
    mod_predict.get_normalized_data()
    masterdf[pred_col] = mod_predict.get_model_predictions()
    masterdf.to_csv(datapath['modelresultspath'], index=False)
    return


##################Part to run the optimization to select the playing 11#################################################
def execute_team_selection(datapath, constconfig, colconfig):
    """

    :return:
    """
    ipl_features = pd.read_csv(datapath['modelresultspath'])
    # selecting the 11 players from a team of 22 based on historic points average

    # get the team by running binary LP solver
    optimum_team = SelectPlayingTeam(ipl_features, constconfig, colconfig)
    # select Top11 based on the predicted points
    optimum_team.select_top11_players(pointscol=colconfig['PREDPOINTS'], selectioncol=colconfig['PREDSELECTION'],
                                      rankcol=colconfig['PREDSELECTIONRANK'], adjustcappoints=True)
    return optimum_team

##################Part to calculate the accuracy of the selected 11 if the actual 11 is available#######################
def execute_rewards_calcualtion(datapath, constconfig, colconfig, rewardconfig):
    """

    :return:
    """
    matchdata = pd.read_csv(datapath['matchdatapath'])
    optimum_team = execute_team_selection(datapath, constconfig, colconfig)
    SQUADCOUNT = 11

    # select Top11 based on the actual points
    optimum_team.select_top11_players(pointscol=colconfig['ACTUALPOINTS'], selectioncol=colconfig['ACTUALSELECTION'],
                                      rankcol=colconfig['ACTUALSELECTIONRANK'], adjustcappoints=True)

    optimum_team.team_points.to_csv(r'Data/team_points.csv', index=False)
    # get the rewards estimate
    ipl_team_rewards = RewardEstimate(optimum_team.team_points, matchdata.copy())

    # get the percentile of the predicted team vs actual tam
    ipl_team_rewards.compare_pred_vs_actual_points(minplayercount=SQUADCOUNT)

    # estimating the monetary impact of the project
    ipl_team_rewards.get_estimated_rewards(rewardconfig, fixed_multipler=50)
    ipl_team_rewards.total_match_points.to_csv(r'Data/rewards_df.csv', index=False)

    # calculating a yearly summary of the model
    yearly_summary = ipl_team_rewards.get_rewards_summary()
    yearly_summary.to_csv(r'Data/rewards_yearly_summary.csv', index=False)
    print(ipl_team_rewards.total_match_points['rewards_earned'].sum())


def create_pred_dataframe(datapath, colconfig, team1, team2, city, venue, toss_winner):
    """

    :param team1:
    :param team2:
    :param city:
    :param venue:
    :return:
    """
    matchdatatemp = pd.read_csv(datapath['matchdatascorecardpath'])
    matchsummary = pd.read_csv(datapath['matchsummarypath'])
    matchdatatemp['playing_team'] = np.where(pd.isnull(matchdatatemp['batsmen_innings']), matchdatatemp['bowler_bowlingteam'], matchdatatemp['batsmen_battingteam'])
    matchdatatemp2 = matchdatatemp[(matchdatatemp['playing_team'].isin([team1, team2]))]
    matchid_rank = pd.DataFrame(matchdatatemp2[['matchid', 'playing_team']].groupby('playing_team')['matchid'].rank(ascending=False, method='dense').
                                reset_index().rename(columns={'matchid': 'rank'}))
    matchid_rank_list = matchid_rank[matchid_rank['rank'] <= 2]['index'].unique()
    matchdatatemp2 = matchdatatemp2[matchdatatemp2.index.isin(matchid_rank_list)][['playername', 'playing_team','playing_role','batsmen_innings','bowler_bowlingteam','batsmen_battingteam','batsmen_bowlingteam']]
    matchdatatemp2 = matchdatatemp2.drop_duplicates(subset=['playername', 'playing_team'])
    MATCHID = matchdatatemp['matchid'].max() + 1
    matchdatatemp2['matchid'] = MATCHID
    matchdatatemp = matchdatatemp.append(matchdatatemp2)
    matchdatatemp.to_csv(datapath['predscorecardpath'], index=False)
    summarydf = pd.DataFrame({"matchid": [MATCHID], "venue": [venue], "city": [city], "team1": [team1], "team2": [team2],
                              "year": [2020], "toss_winner": [toss_winner]})
    matchsummary_pred = matchsummary.append(summarydf)
    matchsummary_pred.to_csv(datapath['predsummarypath'], index=False)
    matchdatafeatures = execute_featureengg(datapath['predscorecardpath'], datapath['predsummarypath'], datapath['predfeaturepath'], colconfig)
    matchdatafeatures = matchdatafeatures[matchdatafeatures['matchid'] == MATCHID]
    matchdatafeatures['opposition_team'] = np.where(matchdatafeatures['playing_team'] == team1, team2, team1)
    matchdatafeatures.to_csv(datapath['predfeaturepath'], index=False)
    return


