from data_prep import ScoreCard, Dream11Points, FeatEngineering
from optimized_selection import SelectPlayingTeam,RewardEstimate
import pandas as pd
import numpy as np
import pickle
from point_prediction import ModelTrain, ModelPredict
from download_ipl20 import update_ipl20_master,get_current_squad
from datetime import datetime
import pytz

##################Part to create additional features for modeling#######################################################

def execute_get_scorecard(matchdatapath, scorecardpath,pointsconfig):
    """

    :return:
    """
    # getting the scorecard from a batsmen's perspective
    matchdata = pd.read_csv(matchdatapath)
    ipl_scorecard = ScoreCard(matchdata)
    # merging both the batsmen and bowler's points to get a single view
    ipl_scorecard.merge_player_scorecard()
    # calculating the points scored by the players based on dream11 scoring method
    ipl_scorecard_points = Dream11Points(ipl_scorecard.ipl_merged_scorecard, pointsconfig)
    ipl_scorecard_points.get_batsmen_bowler_points()
    ipl_scorecard_points.player_scorecard.to_csv(scorecardpath,index = False)
    return ipl_scorecard_points.player_scorecard

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
    # writing the featengg to save it
    # Temp till we get better alternate to cost of each player
    # FeatEng.ipl_features['playercost'] = 10
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
    modeltrain.get_normalized_data()
    modeltrain.get_test_train(split_col='year', split_value=[2019])
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
        return masterdf

    modelpkl = pickle.load(open(datapath['modelpath'], 'rb'))
    enc = pickle.load(open(datapath['encoderpath'], 'rb'))
    mod_predict = ModelPredict(masterdf, enc, modelpkl, modelname, predictors, cat_cols, pred_col)
    mod_predict.get_normalized_data()
    masterdf[pred_col] = mod_predict.get_model_predictions()
    masterdf.to_csv(datapath['modelresultspath'], index=False)
    return masterdf


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
                                      rankcol=colconfig['PREDSELECTIONRANK'], adjustcappoints=False)
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

    optimum_team.team_points.to_csv(datapath['teampoints'], index=False)
    # get the rewards estimate
    ipl_team_rewards = RewardEstimate(optimum_team.team_points, matchdata.copy())

    # get the percentile of the predicted team vs actual tam
    ipl_team_rewards.compare_pred_vs_actual_points(minplayercount=SQUADCOUNT)

    # estimating the monetary impact of the project
    ipl_team_rewards.get_estimated_rewards(rewardconfig, fixed_multipler=50)
    ipl_team_rewards.total_match_points.to_csv(datapath['rewardspath'], index=False)

    # calculating a yearly summary of the model
    yearly_summary = ipl_team_rewards.get_rewards_summary()
    yearly_summary.to_csv(datapath['yearlrewardspath'], index=False)
    print(ipl_team_rewards.total_match_points['rewards_earned'].sum())


def create_pred_dataframe_before_playing_XI(datapath, colconfig, team1, team2, city, venue, toss_winner):
    """

    :param team1:
    :param team2:
    :param city:
    :param venue:
    :return:
    """
    matchdatatemp = pd.read_csv(datapath['matchdatascorecardpath'])
    matchsummary = pd.read_csv(datapath['matchsummarypath'])

    ipl_curr_squad = pd.read_csv(datapath['iplcurrentsquad'])
    ipl_curr_squad = ipl_curr_squad[ipl_curr_squad['playing_team'].isin([team1, team2])]

    ipl_curr_squad['opposition_team'] = np.where(ipl_curr_squad['playing_team'] == team1, team2, team1)
    ipl_curr_squad = ipl_curr_squad[['playername', 'playing_team', 'playing_role', 'opposition_team', 'playercost']]
    MATCHID = matchdatatemp['matchid'].max() + 1
    ipl_curr_squad['matchid'] = MATCHID
    print("after importing ipl squad")
    print(ipl_curr_squad)
    matchdatatemp = pd.concat([matchdatatemp, ipl_curr_squad], axis =0)
    matchdatatemp.to_csv(datapath['predscorecardpath'], index=False)
    summarydf = pd.DataFrame({"matchid": [MATCHID], "venue": [venue], "city": [city], "team1": [team1], "team2": [team2],
                              "year": [2020], "toss_winner": [toss_winner]})
    matchsummary_pred = pd.concat([matchsummary, summarydf], axis=0)
    matchsummary_pred.to_csv(datapath['predsummarypath'], index=False)
    matchdatafeatures = execute_featureengg(datapath['predscorecardpath'], datapath['predsummarypath'], datapath['predfeaturepath'], colconfig)
    matchdatafeatures = matchdatafeatures[matchdatafeatures['matchid'] == MATCHID]
    print("after running the feat engg function")
    print(matchdatafeatures[['playing_team','opposition_team']])
    matchdatafeatures.to_csv(datapath['predfeaturepath'], index=False)
    print("Added pred features for the current match")
    return


def create_pred_dataframe_after_playing_XI(datapath):
    """

    :param datapath:
    :return:
    """
    playing_squad = get_current_squad()
    print("shape of playing squad", playing_squad.shape)
    prefeaturedata = pd.read_csv(datapath['predfeaturepath'])
    team_list = prefeaturedata['playing_team'].unique()
    if playing_squad.shape[0] != 0:
        prefeaturedata = prefeaturedata[prefeaturedata['playername'].isin(playing_squad['playername'])]
    else:
        matchdatatemp = pd.read_csv(datapath['matchdatascorecardpath'])
        matchdatatemp['playing_team'] = np.where(pd.isnull(matchdatatemp['batsmen_innings']),matchdatatemp['bowler_bowlingteam'],matchdatatemp['batsmen_battingteam'])
        matchdatatemp2 = matchdatatemp[(matchdatatemp['playing_team'].isin(team_list))]
        matchid_rank = pd.DataFrame(matchdatatemp2[['matchid', 'playing_team']].groupby('playing_team')['matchid'].rank(ascending=False,method='dense').
            reset_index().rename(columns={'matchid': 'rank'}))
        matchid_rank_list = matchid_rank[matchid_rank['rank'] <= 2]['index'].unique()
        matchdatatemp2 = matchdatatemp2[matchdatatemp2.index.isin(matchid_rank_list)][
            ['playername', 'playing_team', 'playing_role', 'batsmen_innings', 'bowler_bowlingteam','batsmen_battingteam', 'batsmen_bowlingteam']]
        matchdatatemp2 = matchdatatemp2.drop_duplicates(subset=['playername', 'playing_team'])
        prefeaturedata = prefeaturedata[prefeaturedata['playername'].isin(matchdatatemp2['playername'])]
    prefeaturedata.to_csv(datapath['predfeaturepath'], index=False)
    print("preddatafeature updated with currently playing members")

    return


def formatdata(finaloutdf):
    """

    :param finaloutdf:
    :return:
    """

    finaloutdf = finaloutdf[['playername','playing_team','playing_role','playercost','pred_points_catboost','pred_selection_true_catboost','pred_points_ensemble','pred_selection_true_ensemble']]
    finaloutdf.columns = ['playername','teamname','playingrole','playercost','model1_points','model1_team','model2_points','model2_team']
    finaloutdf['model1_points'] = finaloutdf['model1_points'].apply(lambda x: round(x, 0))
    finaloutdf['model2_points'] = finaloutdf['model2_points'].apply(lambda x: round(x, 0))
    return finaloutdf

def update_master_data(datapath,pointsconfig,year):
    matchdata_ipl20 = update_ipl20_master(year)
    matchdata = pd.read_csv(datapath['matchdatapath'])
    if matchdata_ipl20.shape[0] != 0:
        matchlistipl20 = matchdata_ipl20['matchid'].unique()
        matchlistoverall = matchdata['matchid'].unique()
        matchid_list = [i for i in matchlistipl20 if i not in matchlistoverall]
        if matchid_list:
            matchdata_ipl20_sub = matchdata_ipl20[matchdata_ipl20['matchid'].isin(matchid_list)]
            matchdata = pd.concat([matchdata, matchdata_ipl20_sub], join='inner', axis=0)
            matchdata.to_csv(datapath['matchdatapath'], index=False)

        # update scorecard
        master_scorecard = pd.read_csv(datapath['matchdatascorecardpath'])
        matchlistipl20 = matchdata_ipl20['matchid'].unique()
        matchlistoverall = master_scorecard['matchid'].unique()
        matchid_list = [i for i in matchlistipl20 if i not in matchlistoverall]
        if matchid_list:
            matchdata_ipl20_sub = matchdata_ipl20[matchdata_ipl20['matchid'].isin(matchid_list)]
            matchdata_ipl20_sub.to_csv(datapath['matchdatapathipl20'], index=False)
            sorecard_sub = execute_get_scorecard(datapath['matchdatapathipl20'], datapath['matchdatascorecardpathipl20'], pointsconfig)
            master_scorecard = pd.concat([master_scorecard, sorecard_sub], join='inner', axis=0)
            print(master_scorecard.columns)
            master_scorecard.to_csv(datapath['matchdatascorecardpath'], index=False)
            print("matchscorecard updated complete")

        matchsummary_ipl20 = pd.read_csv(datapath['matchsummarypathipl20'])
        matchsummary_ipl20 = matchsummary_ipl20[~(matchsummary_ipl20['winner'] == "Match Tied/Cancelled/Not yet ended")]
        matchsummary = pd.read_csv(datapath['matchsummarypath'])
        matchlistipl20 = matchsummary_ipl20['matchid'].unique()
        matchlistoverall = matchsummary['matchid'].unique()
        matchid_list = [i for i in matchlistipl20 if i not in matchlistoverall]
        print(matchid_list)
        if matchid_list:
            print("start of match summary updation")
            matchsum_ipl20_sub = matchsummary_ipl20[matchsummary_ipl20['matchid'].isin(matchid_list)]
            print(matchsum_ipl20_sub.columns)
            print(matchsummary.columns)
            matchsummary = pd.concat([matchsummary, matchsum_ipl20_sub],axis=0)
            matchsummary.to_csv(datapath['matchsummarypath'], index=False)
            print("matchsummary updated complete")

    return matchdata



def get_team_details(datapath,index =0):

    tz_dubai = pytz.timezone('Asia/Dubai')
    datetime_dubai = datetime.now(tz_dubai)
    matchsummary = pd.read_csv(datapath['matchsummarypathipl20'])
    matchid = matchsummary.iloc[next(x[0] for x in enumerate(pd.to_datetime(matchsummary['date']).tolist()) if x[1] > datetime_dubai), 0]
    today_match = matchsummary[matchsummary['matchid'] == matchid]
    print(today_match)
    team1 = today_match['team1'].iloc[index]
    team2 = today_match['team2'].iloc[index]
    venue = today_match['venue'].iloc[index].split(",")[index]
    return team1, team2, venue