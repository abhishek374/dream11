[![Not Maintained](https://img.shields.io/badge/Maintenance%20Level-Not%20Maintained-yellow.svg)](https://gist.github.com/cheerfulstoic/d107229326a01ff0f333a1d3476e068d)
# Dream11
## Repo OverView
Aim is to select the team that would get the maximum total points in an IPL match under dream 11 constraints. To do so we have followed a 2 stage approach, where we first try to predict points achieved by individual players and then selecting the top 11 out of the total squad as per the cost of each player and other relevant constraints. We are using a mix integer linear optimization method to get the team out of the squad. To get the points of each player we are exploring various methods to leverage the past performance of players to predict their points in the following matches. Below is the description of how the modules are structured and the results thus obtained.

## How to Run ?
Clone the repo to your local and run the controller.py file. It will automatically generate the best 11 based on multiple models into a file pred_team11.csv within Data/ folder. If run after the squad is announced for the match and before the match starts, it will slect the best XI from the current playing squad otherwise will refer to older matches to get the playing XI. The dataset Data/ipl_squad_points.csv has the details for ipl 2020 regarding players role and cost as per dream11. 
For any clarification please drop a note at abhishek.anand374@gmail.com. Will try to respond as soon as possible.

## Dataset Descriptions
Data/pred_team11.csv - result file with the best XI from the playing for the match to be played next
###TODO - Rest descriptions to be added

## Code Descriptions
controller.py -  controller code used to define configs and execute the whole code

main.py - Temp filw with helpfer functions to call other modules, will be merged with other classes within an execute function

data_prep.py - ScoreCard- Used to summarize tha ball by ball data into match level scorecard and also define the playing role for each player
	       Dream11Points - Points calculated as per dream 11 rules: ipl_scorecard_points.csv
	       FeatEngg - Feature enginnering module used to add additional features like, opposition team, venue, city, rolling average of batting order, bowls bowled, 		player's batting points earned, player's bowling points earned, venue's batting point earned and venue's baowling point earned

download_ipl20.py - Has two methods update_ipl20_master: to updated the masterdata needed for prediction by appending the scorecard for the all the matches played till date 								in ipl20
				    get_current_squad: gets the playing XI declared for the most recent match to be played, it attempts to get the match by looking through 							   the website 3 times, if the squad is not out then returns null


point_prediction.py - Module with methods to predict players point in a match based on model specified -"xgb","catboost","rf","svm","arima"
		get_points_moving_avg() - Gives a moing average points per player as a predicted column added to the df: output: ipl_scorecard_points_avg.csv
		train_model() - builds the model based on the input modelname and dataset

optimized_selection.py - SelectPlayingTeam: Module to select 11 players out of the full squad based on the dream11 contraints and to get maximum total points from the game 				 based on actual match results and the one we predicted
			 RewardEstimate: compare_pred_vs_actual_points(): Module to estimate percentile of the predicted from the actual maximum points scored in an match
			 		 get_estimated_rewards(): Convert the percentile into monetary rewards to get an estimate : outut: rewards_df.csv
parse_teams.py - independent script to get the squad for all ipl teams for the year 2020					 
			 		
## Current Results: 

### Overall Results 
2019 (Test Data)

| method | Error | Rewards |
|:-----|:-------:|------:|
| Moving Average | 34% | 9,125 |
| XGBoost | 32% | 550 |
| Catboost | 33% | 14400 |
| randomforest | |  |
| ensemble |  |  |

### Tournament Wise Results (Expected Rewards in INR): 

| year | rewards_xgboost | rewards_moving_avg | rewards_catboost |
|:------|:----------:|----------------:|------------------------:|
| 2008 | 35575 | -75            | 2100 |
| 2009 | 1550 | -925           | 1000 |
| 2010 | 6700 | 675            | 6100 |
| 2011 | 17875 | 2300           | 195400 |
| 2012 | 9475 | -675           | 14025 |
| 2013 | 11475 | 3875           | 2200 |
| 2014 | 1550  | -1775          | -350 |
| 2015 | 6435 | 6300           | 7875 |
| 2016 | 3950 | -275           | 26825 |
| 2017 | 15650 | -1600          | 3550 |
| 2018 | 5550 | -850           | 2875 |
| 2019 | 550 | 9125           | 14400 |



Link to Medium article explaining the approach - https://madhavgoswami.medium.com/dream11-team-predictor-with-python-and-machine-learning-f0dfce1489eb

					
			  		


