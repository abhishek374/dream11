# dream11
## Repo OverView
Project to predict the dream11 points in an IPL match
matchdata.csv - Raw dataset with ball by ball summary of all matches in the IPL so far

main.py - controller code used to define configs and call the other modules

data_prep.py - ScoreCard- Used to summarize tha ball by ball data into match level scorecard and also define the playing role for each player
	       Dream11Points - Points calculated as per dream 11 rules: ipl_scorecard_points.csv

point_prediction.py - Module with methods to predict players point in a match:
		get_points_moving_avg() - Gives a moing average points per player as a predicted column added to the df: output: ipl_scorecard_points_avg.csv

optimized_selection.py - SelectPlayingTeam: Module to select 11 players out of the full squad based on the dream11 contraints and to get maximum total points from the game 				 based on actual match results and the one we predicted
			 RewardEstimate: compare_pred_vs_actual_points(): Module to estimate percentile of the predicted from the actual maximum points scored in an match
			 		 get_estimated_rewards(): Convert the percentile into monetary rewards to get an estimate : outut: rewards_df.csv
			 		
Current Results: 

## Moving Average Method-

1. Overall Accuracy Avg: 76%
2. Total Rewards from IPL: INR 16000/-
3. Tournament Wise Results: 

| year | accuracy_moving_avg | rewards_moving_avg |
|:------|:----------:|----------------:|
| 2008 | 0.305532 | -75            |
| 2009 | 0.369222 | -925           |
| 2010 | 0.322313 | 675            |
| 2011 | 0.312871 | 2300           |
| 2012 | 0.350719 | -675           |
| 2013 | 0.331013 | 3875           |
| 2014 | 0.39659  | -1775          |
| 2015 | 0.316104 | 6300           |
| 2016 | 0.312403 | -275           |
| 2017 | 0.371433 | -1600          |
| 2018 | 0.333714 | -850           |
| 2019 | 0.347684 | 9125           |




					
			  		


