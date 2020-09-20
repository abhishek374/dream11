import requests
from lxml import html
import json
import pandas as pd
import os
from bs4 import BeautifulSoup
import re


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36',
    "Upgrade-Insecure-Requests": "1", "DNT": "1",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate"}


eventid_int=range(1216492,1216530)

MI_squad = pd.DataFrame(columns = ['playername','iscaptain', 'position', 'profilelink', 'teamname'])
CSK_squad = pd.DataFrame(columns = ['playername','iscaptain', 'position', 'profilelink', 'teamname'])
RR_squad = pd.DataFrame(columns = ['playername','iscaptain', 'position', 'profilelink', 'teamname'])
DC_squad = pd.DataFrame(columns = ['playername','iscaptain', 'position', 'profilelink', 'teamname'])
SRH_squad = pd.DataFrame(columns = ['playername','iscaptain', 'position', 'profilelink', 'teamname'])
RCB_squad = pd.DataFrame(columns = ['playername','iscaptain', 'position', 'profilelink', 'teamname'])
KXIP_squad = pd.DataFrame(columns = ['playername','iscaptain', 'position', 'profilelink', 'teamname'])
KKR_squad = pd.DataFrame(columns = ['playername','iscaptain', 'position', 'profilelink', 'teamname'])


for eventid in eventid_int : 

    URL= "https://hsapi.espncricinfo.com/v1/pages/match/home?lang=en&leagueId=8048&eventId="+str(eventid)+"&liveTest=false&qaTest=false"
    response = requests.get(URL, headers=headers)
    data = json.loads(response.text)

    for team in data['content']['squads']:
        for playerlist in team['players']:
            dict ={}
            playername = playerlist['name']
            iscaptain = playerlist['isCaptain']
            position = playerlist['position']
            profilelink = playerlist['link']['href']
            teamname = team['teamName']
            dict =  {'playername':playername, 'iscaptain': iscaptain, 'position': position, 'profilelink': profilelink,'teamname':teamname}
            if team['teamName'] == 'MI':
                MI_squad = MI_squad.append(dict,ignore_index= True)
            elif team['teamName'] == 'CSK':
                CSK_squad = CSK_squad.append(dict,ignore_index = True)
            elif team['teamName'] == 'RR':
                RR_squad = RR_squad.append(dict,ignore_index = True)
            elif team['teamName'] == 'DC':
                DC_squad = DC_squad.append(dict,ignore_index = True)
            elif team['teamName'] == 'SRH':
                SRH_squad = SRH_squad.append(dict,ignore_index = True)
            elif team['teamName'] == 'RCB':
                RCB_squad = RCB_squad.append(dict,ignore_index = True)
            elif team['teamName'] == 'KXIP':
                KXIP_squad = KXIP_squad.append(dict,ignore_index = True)
            elif team['teamName'] == 'KKR':
                KKR_squad = KKR_squad.append(dict,ignore_index = True)
                
    

MI_squad.drop_duplicates(subset=None, keep='first', inplace=True)
CSK_squad.drop_duplicates(subset=None, keep='first', inplace=True)
RR_squad.drop_duplicates(subset=None, keep='first', inplace=True)
DC_squad.drop_duplicates(subset=None, keep='first', inplace=True)
SRH_squad.drop_duplicates(subset=None, keep='first', inplace=True)
RCB_squad.drop_duplicates(subset=None, keep='first', inplace=True)
KXIP_squad.drop_duplicates(subset=None, keep='first', inplace=True)
KKR_squad.drop_duplicates(subset=None, keep='first', inplace=True)

directory='~/Documents/GitHub/dream11/ipl20/'
MI_squad.to_csv(directory+'teams/MI_squad.csv',index = False)
CSK_squad.to_csv(directory+'teams/CSK_squad.csv', index=False)
RR_squad.to_csv(directory+'teams/RR_squad.csv', index=False)
DC_squad.to_csv(directory+'teams/DC_squad.csv', index=False)
SRH_squad.to_csv(directory+'teams/SRH_squad.csv', index=False)
RCB_squad.to_csv(directory+'teams/RCB_squad.csv', index=False)
KXIP_squad.to_csv(directory+'teams/KXIP_squad.csv', index=False)
KKR_squad.to_csv(directory+'teams/KKR_squad.csv', index=False)
