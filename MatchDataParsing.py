import requests
from lxml import html
import json
import pandas as pd
import os
from bs4 import BeautifulSoup
import re



# matchids = {"ipl14":range(729279,734050), "ipl15": range(829705,829824), "ipl16" : range(980901,981020),
#             "ipl17":range(1082591,1082650), "ipl18" : range(1136561,1136621), "ipl19": range(1175356,1175368) }

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36',
    "Upgrade-Insecure-Requests": "1", "DNT": "1",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate"}


def parse_commentary_to_data( data, page, eventid, inning, ipl):

    print ("Parsing the data now to fetch relevant information")
    five_ball_summary = pd.DataFrame(columns=['ipl_season','sequence', 'eventid', 'innings', 'target', 'remainingballs', 'homescore', 'awayscore', 'fallofwickets', 'ball', 'over', 'scorevalue', 'validball', 'extras', 'dismissal', 'dismissaltype', 'batsmanid', 'batsman', 'batsmanteam', 'bowlerid', 'bowler', 'bowlerteam', 'otherathleteinvolvedid', 'otherathleteinvolved', 'nonstrikerid', 'nonstriker', 'runs', 'runrate'])

    for x in data['commentary']['items']:

        print (x)
        if not x:
            print ("there is no data in this "+ str(page)+ " and inning "+str(inning))
            break
        else:
            print ("entered to add rows")
            dict = {}
            ipl_season = ipl
            sequence = x['sequence']
            eventid = eventid
            innings = x['periodText']
            ball = x['innings']['balls']
            over = x['over']['overs']
            scorevalue = x['scoreValue']
            validball = x['over']['noBall'] + x['over']['wide']
            extras = x['over']['noBall'] + x['over']['wide'] + x['over']['byes'] + x['innings']['legByes']
            dismissal = x['dismissal']['dismissal']
            dismissaltype = x['dismissal']['type']
            batsmanid = x['batsman']['athlete']['id']
            batsman = x['batsman']['athlete']['name']
            batsmanteam = x['batsman']['team']['name']
            bowlerid = x['bowler']['athlete']['id']
            bowler = x['bowler']['athlete']['name']
            bowlerteam = x['bowler']['team']['name']
            if dismissal == 1:
                otherathleteinvolvedid = x['athletesInvolved'][0]['id']
                otherathleteinvolved = x['athletesInvolved'][0]['name']
            else:
                otherathleteinvolvedid = ''
                otherathleteinvolved = ''
            nonstrikerid = x['otherBatsman']['athlete']['id']
            nonstriker = x['otherBatsman']['athlete']['name']
            runs = x['innings']['totalRuns']
            fallofwickets = x['innings']['fallOfWickets']
            runrate = x['innings']['runRate']
            target = x['innings']['target']
            remainingballs = x['innings']['remainingBalls']
            homescore = x['homeScore']
            awayscore = x['awayScore']
            dict = {'ipl_season':ipl_season, 'sequence': sequence, 'eventid': eventid, 'innings': innings,
                            'target': target, 'remainingballs': remainingballs, 'homescore': homescore,
                            'awayscore': awayscore,
                            'fallofwickets': fallofwickets, 'ball': ball, 'over': over, 'scorevalue': scorevalue,
                            'validball': validball,
                            'extras': extras, 'dismissal': dismissal, 'dismissaltype': dismissaltype,
                            'batsmanid': batsmanid, 'batsman': batsman, 'batsmanteam': batsmanteam,
                            'bowlerid': bowlerid, 'bowler': bowler, 'bowlerteam': bowlerteam,
                            'otherathleteinvolvedid': otherathleteinvolvedid,
                            'otherathleteinvolved': otherathleteinvolved,
                            'nonstrikerid': nonstrikerid, 'nonstriker': nonstriker,
                            'runs': runs, 'runrate': runrate
                            }
            print("Dictionary from this page is :" + str(dict))
            five_ball_summary = five_ball_summary.append( dict , ignore_index=True)
        # print (df.head())
        # Hypothesis 1 = runrate and balls completed should be somewhat related with the required run rate,
        # so cant use these 2 data simultaneously
    return five_ball_summary


def hit_api(tournamentid, eventid,ipl):
    print('Downloading data for eventid : ' + str(eventid))
    matchdata = pd.DataFrame()

    for inning in range(1, 3):
        print('Downloading data for inning : ' + str(inning))
        # This loop should be increased to range(1,5) for test matches
        try:
            for j in range(1, 8):
                # Will need  a better mechanism to see how many pages of data needs to be present for ODIs and
                # test matches
                page = j
                URL = 'https://site.web.api.espn.com/apis/site/v2/sports/cricket/' + str(
                    tournamentid) + '/playbyplay?contentorigin=espn&event=' + str(eventid) + '&page=' + str(
                    page) + '&period=' + str(inning) + '&section=cricinfo'
                # print (URL)
                response = requests.get(URL, headers=headers)
                data = json.loads(response.text)
                print (len(data['commentary']['items']))
                five_ball_summary = parse_commentary_to_data(data, page, eventid, inning,ipl)
                # print (parse_commentary_to_data(data, page, eventid, inning,ipl))
                # print (five_ball_summary.head())
                matchdata = matchdata.append(five_ball_summary)

        except:
            print ("caught an exception while downloading data for below :")
            print(URL)
    return matchdata


def get_data_for_event(tournamentid, eventid,ipl):
    fullpath = os.path.join(directory, str(eventid) + '.csv')
    if not os.path.isfile(fullpath):
        matchdata = hit_api(tournamentid, eventid,ipl)
        matchdata.to_csv(fullpath, index=False, header=True)
    else:

        print (fullpath)

        print (os.stat(fullpath).st_size == 0)

        try :
            old_data = pd.read_csv(fullpath)

            print(old_data.empty)
            if old_data.empty:
            # if file is present but is empty
                matchdata = hit_api(tournamentid, eventid, ipl)
                with open(fullpath, 'w') as f:
                    matchdata.to_csv(f, header=True, index=False)
            else:

                print('Old Data is present till {} ,  and innings : {}'.format(str(old_data.iloc[-1].iloc[9]),
                                                                               str(old_data.iloc[-1].iloc[2])))
                if str(old_data.iloc[-1].iloc[9]) == '19.6' and str(old_data.iloc[-1].iloc[2]) == '2nd innings' and str(
                        old_data.iloc[-1].iloc[9]) == '0.1':
                    print('data for {} eventid is already present'.format(str(eventid)))
                    pass
                else:
                    matchdata = hit_api(tournamentid, eventid, ipl)
                    with open(fullpath, 'w') as f:
                        matchdata.to_csv(f, header=True, index=False)
        except:
            print(" old data is null file, fetching it again")
            matchdata = hit_api(tournamentid, eventid , ipl)
            print(matchdata.head())
            with open(fullpath, 'w') as f:
                matchdata.to_csv(f, header=True, index=False)

    print ("Added data for " +str(eventid))


def get_match_summary(IPL_Series):
    summarycolnames = ('ipl', 'date', 'matchid', 'inningsinfo1', 'inningsinfo2', 'matchstatus','commentary_link')
    # summarycolnames = ('ipl','date','matchid','typeofmatch',
    #                    'firstteam','firstteamscore','secondteam','secondteamscore','matchstatus')

    matchlist = pd.DataFrame(columns=summarycolnames)

    for key in IPL_Series.keys():
        # print(IPL_Series[key])
        response = requests.get(IPL_Series[key], headers=headers)
        # print(response.text)
        soup = BeautifulSoup(response.text, 'html.parser')
        ipl = key

        matchinformation = soup.find_all(class_='default-match-block')
        print (ipl)

        for matchblock in matchinformation:
            # print(matchblock)
            soup2 = BeautifulSoup(str(matchblock), 'html.parser')
            innings_info_1 = soup2.find(class_='innings-info-1').text.strip()
            innings_info_2 = soup2.find(class_='innings-info-2').text.strip()
            date = soup2.find('span', attrs={'class': 'bold'}).text.strip()
            matchstatus = soup2.find(class_='match-status').text.strip()
            link = soup2.find_all(href=True)[0]['href']
            matchid = link.split('/')[6]
            commentary_link = link.replace("scorecard", "commentary")
            matchlist = matchlist.append(
                {'ipl': str(ipl), 'date': str(date), 'eventid': str(matchid), 'inningsinfo1': str(innings_info_1),
                 'inningsinfo2': str(innings_info_2), 'matchstatus': str(matchstatus) ,'commentary_link' : str(commentary_link)}, ignore_index=True)
            # print (matchlist.head())

    return matchlist


# To find IPL 14 Matches
ipl14 = 'http://www.espncricinfo.com/indian-premier-league-2014/engine/match/index/series.html?series=8827'
ipl15 = 'http://www.espncricinfo.com/indian-premier-league-2015/engine/match/index/series.html?series=9657'
ipl16 = 'http://www.espncricinfo.com/indian-premier-league-2016/engine/match/index/series.html?series=11001'
ipl17 = 'http://www.espncricinfo.com/indian-premier-league-2017/engine/match/index/series.html?series=11701'
ipl18 = 'http://www.espncricinfo.com/ci/engine/match/index/series.html?series=12210'
ipl19 = 'http://www.espncricinfo.com/ci/engine/match/index/series.html?series=12741'



IPL_Series = {
#     'ipl14': ipl14,
#     'ipl15': ipl15,
#     'ipl16': ipl16,
#     'ipl17': ipl17,
#     'ipl18': ipl18,
    'ipl19': ipl19
}




if __name__ == '__main__':

    tournamentid = 8048
    path_to_dir = os.getcwd()
    folderloc = os.path.join(os.getcwd(), 'IPL')
    directory = os.path.join(folderloc, str(tournamentid))


    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        os.chdir(directory)

    #
    # matchlist = get_match_summary(IPL_Series)
    # print(matchlist.info())

    # matchlist.to_csv('IPLMatchSummary.csv', index=False, header=True)

    matchsummary = pd.read_csv(directory+'/IPLMatchSummary.csv')
    # print (matchsummary.head())


    def download_event(eventid) :

        print (matchsummary.loc[matchsummary['eventid']==eventid,'ipl'].values[0])
        get_data_for_event(tournamentid, eventid,matchsummary.loc[matchsummary['eventid']==eventid,'ipl'].values[0])

    # download_event (1178409)

    # pd.read_csv("/Users/madhavg/Documents/GitHub/Laptop/game_simulation/IPL/8048/729281.csv").head()

    for row, data in matchsummary.iterrows():
        if data['ipl']=='ipl19':
            print (row)
            eventid = data['eventid']
            print ("downloading data for " +str(eventid))
            get_data_for_event(tournamentid, eventid,matchsummary.loc[matchsummary['eventid']==eventid,'ipl'].values[0])



    #
    # Next Steps :
    # Create Features for Understanding Situations
    # Learn matplotlib doing these things
    # Create heatmaps
    #
    # Find Features and PCA