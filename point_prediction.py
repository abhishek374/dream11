import pandas as pd

class PointPred:


    def get_points_moving_avg(self, ipl_points, rolling_avg_window) -> pd.DataFrame:
        """
        function to calculate predicted points per player based on moving average method
        :param ipl_points: dataframe with the historical points achieved by the player
        :param rolling_avg_window: number of matches to take the rolling average over
        :return: ipl_points: dataframe with the columns total_points_avg, metric used to select the eventual team
        """
        ipl_points = ipl_points.sort_values(by=['matchid', 'playername'], ascending=True)
        ipl_points.set_index('matchid', inplace=True)
        player_avg_points = pd.DataFrame(ipl_points.groupby(['playername'])['total_points'].rolling(rolling_avg_window).mean()).reset_index(). \
            rename(columns={'total_points': 'total_points_avg'})
        player_avg_points = player_avg_points.sort_values(by=['matchid', 'playername'], ascending=True)
        player_avg_points.set_index('matchid', inplace=True)
        player_avg_points['total_points_avg'] = pd.DataFrame(player_avg_points.groupby(['playername'])['total_points_avg'].shift(1))
        ipl_points.reset_index(inplace=True)
        player_avg_points.reset_index(inplace=True)
        ipl_points = pd.merge(ipl_points, player_avg_points, on=['matchid', 'playername'], how='left')
        return ipl_points

