from data_prep import ScoreCard, Dream11Points, FeatEngineering
from optimized_selection import SelectPlayingTeam,RewardEstimate
import pandas as pd
import pickle
from point_prediction import ModelTrain, ModelPredict

##################Part to create additional features for modeling#######################################################

def execute_get_scorecard():
    """

    :return:
    """
    # getting the scorecard from a batsmen's perspective
    ipl_scorecard = ScoreCard(matchdata.copy())
    # merging both the batsmen and bowler's points to get a single view
    ipl_scorecard.merge_player_scorecard()
    # calculating the points scored by the players based on dream11 scoring method
    ipl_scorecard_points = Dream11Points(ipl_scorecard.ipl_merged_scorecard, pointsconfig)
    ipl_scorecard_points.get_batsmen_bowler_points()
    # writing the scorecard to ipl_scorecard_points.csv
    # ipl_scorecard_points.player_scorecard.to_csv(r'Data/ipl_scorecard_points.csv', index=False)

    ##################Part to create additional features for modeling#######################################################

    FeatEng = FeatEngineering(ipl_scorecard_points.player_scorecard, matchsummary)
    FeatEng.add_venue_info()
    FeatEng.add_homegame_flag()
    FeatEng.add_toss_info()
    FeatEng.add_player_match_count()

    rolling_window = [2, 5, 10]
    for i in rolling_window:
        FeatEng.add_lagging_feat(colconfig['MATCHID'], colconfig['VENUE'], i, colconfig['TOTALBATPOINTS'], colconfig['TOTALBALLPOINTS'])
        FeatEng.add_lagging_feat(colconfig['MATCHID'], colconfig['PLAYERNAME'], i, colconfig['ACTUALPOINTS'], colconfig['TOTALBATPOINTS'], colconfig['TOTALBALLPOINTS'], colconfig['BATTINGORDER'], 'total_balls_bowled')
    # writing the scorecard to save it
    # Temp till we get better alternate to cost of each player
    FeatEng.ipl_features['playercost'] = 10
    FeatEng.ipl_features.to_csv(r'Data/ipl_scorecard_points_featengg.csv', index=False)
    return

##################Part to train the model #################################################
def execute_model_train():
    """

    :return:
    """
    masterdf = pd.read_csv(r'Data/ipl_scorecard_points_featengg.csv')
    modeltrain = ModelTrain(masterdf, target_col, predictors, cat_cols)
    modeltrain.get_normalized_data()
    modeltrain.get_test_train(split_col='year', split_value=[2019])
    # modeltrain.train_model(model='xgb')
    modeltrain.train_model(model='catboost')
    modelobjects = modeltrain.get_model_objects(model='catboost')

    pickle.dump(modelobjects[1], open(modelpath, 'wb'))
    pickle.dump(modelobjects[0], open(encoderpath, 'wb'))
    print(modeltrain.feat_imp_df)

    return

##################Part to train the model #################################################
def execute_model_prediction():
    """

    :return:
    """
    masterdf = pd.read_csv(r'Data/ipl_scorecard_points_featengg.csv')
    modelpkl = pickle.load(open(modelpath, 'rb'))
    enc = pickle.load(open(encoderpath, 'rb'))
    mod_predict = ModelPredict(masterdf, enc, modelpkl, predictors, cat_cols, pred_col)
    mod_predict.get_normalized_data()
    masterdf[pred_col] = mod_predict.get_model_predictions()

    modelresults = mod_predict.get_model_error(masterdf, pred_col, target_col, groupbycol='year')

    masterdf.to_csv(r'Data\model_prediction.csv', index=False)
    print('mean_squared_error: {}'.format(modelresults[0]))
    print('yearly summary mean_squared_error: {}'.format(modelresults[1]))
    return


##################Part to run the optimization to select the playing 11#################################################
def execute_team_selection():
    """

    :return:
    """
    ipl_features = pd.read_csv(r'Data\model_prediction.csv')
    # selecting the 11 players from a team of 22 based on historic points average

    # get the team by running binary LP solver
    optimum_team = SelectPlayingTeam(ipl_features, constconfig, colconfig)
    # select Top11 based on the predicted points
    optimum_team.select_top11_players(pointscol=colconfig['PREDPOINTS'], selectioncol=colconfig['PREDSELECTION'],
                                      rankcol=colconfig['PREDSELECTIONRANK'], adjustcappoints=True)
    return optimum_team

##################Part to calculate the accuracy of the selected 11 if the actual 11 is available#######################
def execute_rewards_calcualtion():
    """

    :return:
    """
    optimum_team = execute_team_selection()
    SQUADCOUNT = 11

    # select Top11 based on the actual points
    # ipl_features = pd.read_csv(r'Data\model_prediction.csv')
    # optimum_team = SelectPlayingTeam(ipl_features, constconfig, colconfig)
    optimum_team.select_top11_players(pointscol=colconfig['ACTUALPOINTS'], selectioncol=colconfig['ACTUALSELECTION'],
                                      rankcol=colconfig['ACTUALSELECTIONRANK'], adjustcappoints=True)

    optimum_team.team_points.to_csv(r'Data/team_points.csv')
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

if __name__ == "__main__":
    # reading the source file from local
    matchdata = pd.read_csv(r'Data/matchdata.csv')
    matchsummary = pd.read_csv(r'Data/matchsummary.csv')
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

    predictors = ['playing_team', 'playing_role', 'home_game', 'toss_flag', 'player_match_count',
                  'total_bat_points_playername_avg10',
                  'total_bowl_points_playername_avg10', 'fallofwickets_playername_avg10',
                  'total_balls_bowled_playername_avg10',
                  'total_bat_points_venue_avg10', 'total_bowl_points_venue_avg10', 'total_bat_points_playername_avg5',
                  'total_bowl_points_playername_avg5', 'fallofwickets_playername_avg5',
                  'total_balls_bowled_playername_avg5',
                  'total_bat_points_venue_avg5', 'total_bowl_points_venue_avg5', 'total_bat_points_playername_avg2',
                  'total_bowl_points_playername_avg2', 'fallofwickets_playername_avg2',
                  'total_balls_bowled_playername_avg2',
                  'total_bat_points_venue_avg2', 'total_bowl_points_venue_avg2']
    cat_cols = ['playing_team', 'playing_role']

    pred_col = 'pred_points'
    target_col = 'total_points'
    # modelpath = r"Data\xgb_model.pkl"
    # encoderpath = r"Data\OnHotEncoder_xgb.pkl"
    modelpath = r"Data\catb_model.pkl"
    encoderpath = r"Data\OnHotEncoder_catb.pkl"

    execute_get_scorecard()  # Run the function to to feature engineering
    execute_model_train()  # Run the function to build the model
    execute_model_prediction()  #Rung ht function to predict the points based on the model
    execute_team_selection()  # Run the select the playing 11
    execute_rewards_calcualtion()  # Run the function to estimate rewards

    #TODO make the constraint for allrounder, batsmen, bowler dynamic
