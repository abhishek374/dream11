import requests
import json
import pandas as pd
import os
from datetime import datetime
import pytz
import time


def parse_commentary_to_data(data, page, eventid, inning, ipl):

    #     print ("Parsing the data now to fetch relevant information")
    five_ball_summary = pd.DataFrame(columns=['ipl_season', 'sequence', 'eventid', 'innings', 'target', 'remainingballs', 'homescore', 'awayscore', 'fallofwickets', 'ball', 'over', 'scorevalue', 'validball', 'extras',
                                     'dismissal', 'dismissaltype', 'batsmanid', 'batsman', 'batsmanteam', 'bowlerid', 'bowler', 'bowlerteam', 'otherathleteinvolvedid', 'otherathleteinvolved', 'nonstrikerid', 'nonstriker', 'runs', 'runrate'])

    for x in data['commentary']['items']:

        if not x:
    #             print ("there is no data in this "+ str(page)+ " and inning "+str(inning))
            break
        else:
    #             print ("entered to add rows")
            dict = {}
            ipl_season = ipl
            sequence = x['sequence']
            eventid = eventid
            innings = x['periodText']
            ball = x['innings']['balls']
            over = x['over']['overs']
            scorevalue = x['scoreValue']
            validball = x['over']['noBall'] + x['over']['wide']
            extras = x['over']['noBall'] + x['over']['wide'] + \
                x['over']['byes'] + x['innings']['legByes']
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
            dict = {'ipl_season': ipl_season, 'sequence': sequence, 'eventid': eventid, 'innings': innings,
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
#             print("Dictionary from this page is :" + str(dict))
            five_ball_summary = five_ball_summary.append(
                dict, ignore_index=True)
        # print (df.head())
        # Hypothesis 1 = runrate and balls completed should be somewhat related with the required run rate,
        # so cant use these 2 data simultaneously
    return five_ball_summary


def hit_api(tournamentid, eventid, headers, ipl):
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
#                 print (len(data['commentary']['items']))
                five_ball_summary = parse_commentary_to_data(
                    data, page, eventid, inning, ipl)
                # print (parse_commentary_to_data(data, page, eventid, inning,ipl))
                # print (five_ball_summary.head())
                matchdata = matchdata.append(five_ball_summary)

        except:
            print("caught an exception while downloading data for below :")
            print(URL)
    return matchdata


def get_data_for_event(tournamentid, eventid, directory, headers, ipl):
    fullpath = os.path.join(directory, str(eventid) + '.csv')

    matchdata = hit_api(tournamentid, eventid,headers,ipl )
    matchdata.to_csv(fullpath, index=False, header=True)
    return matchdata


def update_ipl20_master(year):
    directory = os.getcwd() +"/ipl20"
    tournamentid = "8048"
    year= str(year)
    #directory = '~/Documents/GitHub/dream11/ipl20'

    ipl20_schedule_url = "https://hsapi.espncricinfo.com/v1/pages/series/schedule?lang=en&leagueId="+tournamentid+"&year="+year
    headers = {
	    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36',
	    "Upgrade-Insecure-Requests": "1", "DNT": "1",
	    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.5",
	    "Accept-Encoding": "gzip, deflate"
        }
    response = requests.get(ipl20_schedule_url, headers=headers)
    data = json.loads(response.text)
    match_summary_ipl20 = pd.DataFrame(columns=['matchid', 'date', 'year', 'city', 'venue','team1', 'team2', 'result', 'tossdecision', 'winner', 'by_what', 'by_how_many', 'mom'])
    print("data",data)
    for match_details in data['content']['matchEvents']:
	    dict = {}
	    matchid = match_details['id']
	    date = match_details['date']
	    city = match_details['venue']['name']
	    venue = match_details['venue']['name']
	    team1 = match_details['competitors'][0]['name']
	    team2 = match_details['competitors'][1]['name']
	    tossdecision = ""
	    result = match_details['statusText']
	    by_what = ""
	    by_how_many = 0
	    mom = ""
	    winner = ""

	    if match_details['competitors'][0]['isWinner'] == True:
    	#         print(match_details['competitors'][1]['isWinner'])
	        winner = match_details['competitors'][0]['name']
	    elif match_details['competitors'][1]['isWinner'] == True:
	        winner = match_details['competitors'][1]['name']
	    else:
	        winner = 'Match Tied/Cancelled/Not yet ended'

	    if 'runs' in match_details['statusText']:
	        by_what = 'runs'
	    elif 'wickets' in match_details['statusText']:
	        by_what = 'wickets'
	    else:
	        by_what = ''

	    by_how_many = [int(i) for i in match_details['statusText'].split() if i.isdigit()]
	    dict = {'matchid': matchid, 'date': date, 'year': year, 'city': city, 'venue': venue, 'team1': team1, 'team2': team2, 'result': result,'tossdecision': tossdecision, 'winner': winner, 'by_what': by_what, 'by_how_many': by_how_many, 'mom': mom}
	    match_summary_ipl20 = match_summary_ipl20.append(dict, ignore_index=True)

    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # else:
    #     os.chdir(directory)

    match_summary_ipl20.to_csv(directory+'/match_summary_ipl20.csv', index=False)

    match_summary_ipl20 = pd.read_csv(directory+'/match_summary_ipl20.csv')

    # matchdata_full = pd.DataFrame(columns=['ipl_season', 'sequence', 'eventid', 'innings', 'target', 'remainingballs', 'homescore', 'awayscore', 'fallofwickets', 'ball', 'over', 'scorevalue', 'validball', 'extras',
    #                                                             'dismissal', 'dismissaltype', 'batsmanid', 'batsman', 'batsmanteam', 'bowlerid', 'bowler', 'bowlerteam', 'otherathleteinvolvedid', 'otherathleteinvolved', 'nonstrikerid', 'nonstriker', 'runs', 'runrate'])

    matchdata_full = pd.read_csv(directory+'/matchdata_ipl20.csv')
    # matchdata_v2 = pd.DataFrame(columns=['date', 'matchid', 'innings', 'target', 'fallofwickets', 'ball', 'over', 'scorevalue',
    #              'validball', 'extras', 'extratype', 'batsmanname', 'batsmanscorevalue', 'bowlername', 'nonstrikername',
    #              'totalruns', 'dismissal', 'dismissedtype', 'dismissedplayer', 'battingteam', 'bowlingteam'])
    matchdata_v2 = pd.read_csv(directory+'/matchdata_v2.csv')

    for eventid in match_summary_ipl20[~match_summary_ipl20['result'].str.contains('Starts') & match_summary_ipl20['result'].str.contains('won')].matchid:
        print(eventid)
        if eventid not in matchdata_full.eventid.unique().tolist():
            _match_data_ = get_data_for_event(tournamentid, eventid,directory,headers,'ipl20')
            matchdata_full = matchdata_full.append(_match_data_, ignore_index=True,  sort=True)
        else:
            print('data already downloaded')


    matchdata_full.to_csv(directory+'/matchdata_ipl20.csv', index = False)
    ipl20_matchdata = pd.read_csv(directory+'/matchdata_ipl20.csv')

    names_mapping = pd.read_csv(directory+'/name_mapping_clean.csv')
    eventids_to_be_backfilled = [x for x in ipl20_matchdata.eventid.unique().tolist() if x not in matchdata_v2.matchid.unique().tolist()]
    print("matchdata_v2 needs to be updated for ")
    print(eventids_to_be_backfilled)

    for iter, row in ipl20_matchdata[ipl20_matchdata['eventid'].isin(eventids_to_be_backfilled)].iterrows():
        print("row",row)
        print("row['eventid']",row['eventid'])
        date = pd.to_datetime(match_summary_ipl20[match_summary_ipl20.matchid == row['eventid']].date).dt.date
        matchid = row['eventid']
        innings = row['innings']
        target = row['target']
        fallofwickets = row['fallofwickets']
        ball = row['ball']
        over = row['over']
        scorevalue = row['scorevalue']
        validball = row['validball']
        extras = row['extras']
        extratype = 'Nan'
        batsmanscorevalue = row['scorevalue']

        if row['batsman'] in names_mapping.values:
            batsmanname = names_mapping[names_mapping.ipl20_name ==
                                        row['batsman']]['old_name'].tolist()[0]
        else:
            batsmanname = row['batsman']

        if row['bowler'] in names_mapping.values:
            bowlername = names_mapping[names_mapping.ipl20_name ==
                                row['bowler']]['old_name'].tolist()[0]
        else:
            bowlername = row['bowler']

        if row['nonstriker'] in names_mapping.values:
            nonstrikername = names_mapping[names_mapping.ipl20_name ==
                                    row['nonstriker']]['old_name'].tolist()[0]
        else:
            nonstrikername = row['nonstriker']

        if row['homescore'] == 0:
            totalruns = row['awayscore'].split('/')[0]
        else:
            totalruns = row['homescore'].split('/')[0]

        dismissal = row['dismissal']
        dismissedtype = row['dismissaltype']

        if row['dismissal'] == True:
            dismissedplayer = batsmanname
        else:
            dismissedplayer = ''

        battingteam = row['batsmanteam']
        bowlingteam = row['bowlerteam']

        dict = {"date": date, "matchid": matchid, "innings": innings, "target": target, "fallofwickets": fallofwickets,
                "ball": ball, "over": over, "scorevalue": scorevalue, "validball": validball, "extras": extras,
                "extratype": extratype, "batsmanname": batsmanname, "batsmanscorevalue": batsmanscorevalue,
                "bowlername": bowlername, "nonstrikername": nonstrikername, "totalruns": totalruns, "dismissal": dismissal,
                "dismissedtype": dismissedtype, "dismissedplayer": dismissedplayer, "battingteam": battingteam,
                "bowlingteam": bowlingteam}
        matchdata_v2 = matchdata_v2.append(dict, ignore_index=True)

    matchdata_v2.to_csv(directory+'/matchdata_v2.csv', index=False)
    matchdata_v2 = pd.read_csv(directory+'/matchdata_v2.csv')
    return matchdata_v2


def get_current_squad():

    directory = os.getcwd() +"/ipl20"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36',
        "Upgrade-Insecure-Requests": "1", "DNT": "1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate"
    }

    match_summary_ipl20 = pd.read_csv(directory + '/match_summary_ipl20.csv')

    tz_dubai = pytz.timezone('Asia/Dubai')
    datetime_dubai = datetime.now(tz_dubai)

    eventid = match_summary_ipl20.iloc[next(x[0] for x in enumerate(pd.to_datetime(match_summary_ipl20['date']).tolist()) if x[1] > datetime_dubai), 0]

    print('Playing XI need to be downloaded for : ')
    print(eventid)

    today_squad = pd.DataFrame(columns = ['playername','iscaptain', 'position', 'profilelink', 'teamname'])
    squad_title = 'Squads'

    counter = 0
    while squad_title != 'Playing XI' and counter < 3:
        URL= "https://hsapi.espncricinfo.com/v1/pages/match/home?lang=en&leagueId=8048&eventId="+str(eventid)+"&liveTest=false&qaTest=false"
        print(URL)
        response = requests.get(URL, headers=headers)
        data = json.loads(response.text)
        squad_title = data['content']['squads'][0]['title']

        if squad_title == 'Playing XI':
            for team in data['content']['squads']:
                for playerlist in team['players']:
                    dict ={}
                    playername = playerlist['name']
                    iscaptain = playerlist['isCaptain']
                    position = playerlist['position']
                    profilelink = playerlist['link']['href']
                    teamname = team['teamName']
                    dict =  {'playername':playername, 'iscaptain': iscaptain, 'position': position, 'profilelink': profilelink,'teamname':teamname}
                    today_squad = today_squad.append(dict, ignore_index=True)

            today_squad.drop_duplicates(subset=None, keep='first', inplace=True)
            names_mapping = pd.read_csv(directory + '/name_mapping_clean.csv')
            names_mapping.columns = ['playername', 'new_playername']
            today_squad = pd.merge(today_squad, names_mapping, on='playername', how='inner')
            today_squad.drop(columns=['playername'], inplace=True)
            today_squad.rename(columns={'new_playername': "playername"}, inplace=True)
            today_squad.to_csv(directory+'/teams/'+str(eventid)+'_squad.csv', index=False)

        else:
            print('could not find the playing XI on page, trying again in 60 secs')
            time.sleep(60)
        counter = counter + 1
    return today_squad