# dream11
Project to predict the dream11 points in an IPL match
Point Estimation in an IPL Match

Objective: We want to select a team of 11 players out of a squad of 22 players such that it maximizes the Dream 11 points scored in a game
	Maximize: Total Points =  (∑_(i=1)^22▒〖Player_i  * RunShare_i*TotalRuns*RunPoints_i+ Player_i  * WicketsShare_i*TotalWickets*WicketsPoints_i 〗)(N_i)  			(1)
  
Such That –
	Total cost of purchasing the players - 		
	∑_(i=1)^22▒〖Player_i* Cost_i<MAXCOSTLIMIT〗					(2)	
	Constraint on the type of players we can select: Batsmen, All Rounders and Bowlers
	MINBATSMEN≤∑_(i=1)^22▒〖Player_i* Batsmen_i≤MAXBATSMEN〗			(3)
	MINALLROUDNERS≤∑_(i=1)^22▒〖Player_i* Allrounder_i<MAXALLROUNDERS〗	(4)	
	MINBOWLERS≤∑_(i=1)^22▒〖Player_i* 〖Bowler〗_i<MAXBOWLERS〗			(5)
	N_i=2 if Player_i has max⁡〖points,N_i=1〗.5 if Player_i has 2nd highest points othwerwise N_i=1 				(6)
	RunPoints_i=Blended points per run scored based on historical strike rate                       (7)
	WicketsPoints_i=Blended points per wickets taken based on historical economy rate 		(8)

One way to do this is solve it using a mix integer linear programming method. Slightly complicated and take time to code, however we can try to run a brute force algorithm and try to approximate the results also. In that case maximum we will have to run C(22,11) (705,432 iterations). To estimate the objective function we also need to also estimate 
	RunShare_i  =  〖e 〗^(RunPotential_i )/(∑_(i=1)^22▒e^(RunPotential_i ) )							(9)
	WicketsShare_i  =  e^(WicketPotential_i )/(∑_(i=1)^22▒e^(WicketPotential_i ) )						(10)
	RunPotential_i= ∑_j▒〖γ_j*PlayerFeature_j 〗					(11)
	WicketPotential_i= ∑_j▒〖µ_j*PlayerFeature_j 〗					(12)
	To estimate the run potential and wicket potential of a player we can use the historical performance of a player, batting position and impact of other players in the team, 
	TotalRuns= ∑_j▒〖α_j*RunFeature_j 〗						(13)
	TotalWickets= ∑_j▒〖β_j*WicketFeature_j 〗					(14)	
(Total runs scored and total wickets in a match dependent on features such as the historical average of runs scored on that ground, teams playing, others)
Batsmen	Score
Run	1
Boundary Bonus	1
Six Bonus	2
Half-century Bonus	8
Century Bonus	16
Dismissal for a duck	-2
	
Bowler	
WicketExcluding Run Out	25
4 wicket haul Bonus	8
5 wicket haul Bonus	16
Maiden over	8
	
Fielding	
Catch	8
Stumping/Run-out	12
Run Out (Thrower)	8
Run Out (Catcher)	4
	
	
Others	
Captain	2x
Vice-Captain	1.5x
	
	
Economy Rate Points (Min 10 Balls)	
Below 4 runs per over	6
Between 4-4.99 runs per over	4
Between 5-6 runs per over	2
Between 9-10 runs per over	-2
Between 10.1-11 runs per over	-4
Above 11 runs per over	-6
	
	
Strike Rate	
Between 60-70 runs per 100 balls	-2
Between 50-59.9 runs per 100 balls	-4
Below 50 runs per 100 balls	-6

