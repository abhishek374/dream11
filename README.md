# Dream11
## Repo OverView
Aim is to select the team that would get the maximum total points in an IPL match under dream 11 constraints. To do so we have followed a 2 stage approach, where we first try to predict points achieved by individual players and then selecting the top 11 out of the total squad as per the cost of each player and other relevant constraints. We are using a mix integer linear optimization method to get the team out of the squad. To get the points of each player we are exploring various methods to leverage the past performance of players to predict their points in the following matches. Below is the description of how the modules are structured and the results thus obtained.

### Code Descriptions
matchdata.csv - Raw dataset with ball by ball summary of all matches in the IPL so far

main.py - controller code used to define configs and call the other modules

data_prep.py - ScoreCard- Used to summarize tha ball by ball data into match level scorecard and also define the playing role for each player
	       Dream11Points - Points calculated as per dream 11 rules: ipl_scorecard_points.csv

point_prediction.py - Module with methods to predict players point in a match:
		get_points_moving_avg() - Gives a moing average points per player as a predicted column added to the df: output: ipl_scorecard_points_avg.csv

optimized_selection.py - SelectPlayingTeam: Module to select 11 players out of the full squad based on the dream11 contraints and to get maximum total points from the game 				 based on actual match results and the one we predicted
			 RewardEstimate: compare_pred_vs_actual_points(): Module to estimate percentile of the predicted from the actual maximum points scored in an match
			 		 get_estimated_rewards(): Convert the percentile into monetary rewards to get an estimate : outut: rewards_df.csv
			 		
## Current Results: 

### Overall Results 
2019 (Test Data)

| method | Error | Rewards |
|:-----|:-------:|------:|
| Moving Average | 34% | 9,125 |
| XGBoost | 32% | 550 |
| Catboost | 32% | 2675 |

### Tournament Wise Results (Expected Rewards in INR): 

| year | rewards_xgboost | rewards_moving_avg | rewards_catboost |
|:------|:----------:|----------------:|------------------------:|
| 2008 | 35575 | -75            | 33675
| 2009 | 1550 | -925           | 156225 |
| 2010 | 6700 | 675            | 3800 |
| 2011 | 17875 | 2300           | 285925 |
| 2012 | 9475 | -675           | 22350 |
| 2013 | 11475 | 3875           | 38025 |
| 2014 | 1550  | -1775          | 1750 |
| 2015 | 6435 | 6300           | 20750 |
| 2016 | 3950 | -275           | 12850 |
| 2017 | 15650 | -1600          | 152300 |
| 2018 | 5550 | -850           | 6475 |
| 2019 | 550 | 9125           | 2675 |




					
			  		


