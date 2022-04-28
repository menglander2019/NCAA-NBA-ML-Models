import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
import sys
import string
import requests
import datetime
import progressbar
import time
import re


def player_basic_info():
	players = []
	base_url = 'http://www.basketball-reference.com/players/'
	for letter in string.ascii_lowercase:
	    page_request = requests.get(base_url + letter)
	    soup = BeautifulSoup(page_request.text,"lxml")
	    table = soup.find('table')
	    if table:
	        table_body = table.find('tbody')
	        for row in table_body.findAll('tr'):
	            player_url = row.find('a')
	            player_names = player_url.text
	            player_pages = player_url['href']
	            cells = row.findAll('td') # all data for all players uniform across database
	            active_from = int(cells[0].text)
	            active_to = int(cells[1].text)
	            position = cells[2].text
	            height = cells[3].text
	            weight = cells[4].text
	            birth_date = cells[5].text
	            college = cells[6].text    
	            player_entry = {'url': player_pages,
	                            'name': player_names,
	                            'active_from': active_from,
	                            'active_to': active_to,
	                            'position': position,
	                            'college': college,
	                            'height': height,
	                            'weight': weight,
	                            'birth_date': birth_date}
	            players.append(player_entry)
	return pd.DataFrame(players)

def player_info(url):
	#define all quantites
	fgpct = None
	games = None
	ppg = None
	ft = None
	fgpg = None
	fgapg = None
	ftpg = None
	ftapg = None
	_3ptpg = None
	_3ptapg = None
	_3ptpct = None
	efgpct = None
	plyr_eff_rtg = None
	vorp = None
	bpm = None
	ws_per_48 = None
	ws = None
	NCAA_fgpct = None
	NCAA_games = None
	NCAA_ppg = None
	NCAA_ft = None
	NCAA_fgpg = None
	NCAA_fgapg = None
	NCAA_ftpg = None
	NCAA_ftapg = None
	NCAA__3ptpg = None
	NCAA__3ptapg = None
	NCAA__3ptpct = None
	NCAA_efgpct = None
	#print('url = ' + str('http://www.basketball-reference.com' + str(url)))
	page_request = requests.get('http://www.basketball-reference.com' + str(url))
	if page_request.status_code >= 200 and page_request.status_code <= 299:
		soup = BeautifulSoup(page_request.text,"lxml")
		table = soup.find('table') #the first table is luckily the per game stats
		if table:
			table_foot = table.find('tfoot')
			if table_foot:
				for row in table_foot.findAll('tr'):
					cells  = row.findAll('td')			
					playerData = str(cells) #the indexes are not uniform across the database
					#games = re.search(r'data-stat="g">(.*?)</td>', playerData).group(1) # don't need
					fgpct = re.search(r'data-stat="fg_pct">(.*?)</td>', playerData).group(1)
					games = re.search(r'data-stat="g">(.*?)</td>', playerData).group(1)
					ppg = re.search(r'data-stat="pts_per_g">(.*?)</td>', playerData).group(1)
					ft = re.search(r'data-stat="ft_pct">(.*?)</td>', playerData).group(1)
					fgpg = re.search(r'data-stat="fg_per_g">(.*?)</td>', playerData).group(1)
					fgapg = re.search(r'data-stat="fga_per_g">(.*?)</td>', playerData).group(1)
					ftpg = re.search(r'data-stat="ft_per_g">(.*?)</td>', playerData).group(1)
					ftapg = re.search(r'data-stat="fta_per_g">(.*?)</td>', playerData).group(1)
					if re.search(r'data-stat="fg3_per_g">(.*?)</td>', playerData) != None:
						_3ptpg = re.search(r'data-stat="fg3_per_g">(.*?)</td>', playerData).group(1)
					else:
						_3ptpg = None
					if re.search(r'data-stat="fg3a_per_g">(.*?)</td>', playerData) != None:
						_3ptapg = re.search(r'data-stat="fg3a_per_g">(.*?)</td>', playerData).group(1)
					else:
						_3ptapg = None
					if re.search(r'data-stat="fg3_pct">(.*?)</td>', playerData) != None:
						_3ptpct = re.search(r'data-stat="fg3_pct">(.*?)</td>', playerData).group(1)	
					else:
						_3ptpct = None
					if re.search(r'data-stat="efg_pct">(.*?)</td>', playerData) != None:
						efgpct = re.search(r'data-stat="efg_pct">(.*?)</td>', playerData).group(1)
					else:
						efgpct = None
					break  #bad but I want the structure to remain the same in case I want more data outside overall stats


		advanced_table = soup.find(id='all_advanced-playoffs_advanced')
		if advanced_table:
			advanced_table_foot = advanced_table.find('tfoot')
			if advanced_table_foot:
				for row in advanced_table_foot.findAll('tr'):
					cells  = row.findAll('td')			
					playerData = str(cells)
					if re.search(r'data-stat="per">(.*?)</td>', playerData) != None:
						plyr_eff_rtg = re.search(r'data-stat="per">(.*?)</td>', playerData).group(1)
					else:
						plyr_eff_rtg = None
					if re.search(r'data-stat="vorp">(.*?)</td>', playerData) != None:
						vorp = re.search(r'data-stat="vorp">(.*?)</td>', playerData).group(1)
					else:
						vorp = None
					if re.search(r'data-stat="bpm">(.*?)</td>', playerData) != None:
						bpm = re.search(r'data-stat="bpm">(.*?)</td>', playerData).group(1)
					else:
						bpm = None
					if re.search(r'data-stat="ws_per_48">(.*?)</td>', playerData) != None:
						ws_per_48 = re.search(r'data-stat="ws_per_48">(.*?)</td>', playerData).group(1)
					else:
						ws_per_48 = None
					if re.search(r'data-stat="ws">(.*?)</td>', playerData) != None:
						ws = re.search(r'data-stat="ws">(.*?)</td>', playerData).group(1)
					else:
						ws = None
					break



		college_url = get_player_college_url(url)
		if(college_url != None):
			page_request_cbb = requests.get(college_url)
			soupy = BeautifulSoup(page_request_cbb.text,'lxml')
			table_cbb = soupy.find('table')
			if table_cbb:
				table_foot = table_cbb.find('tfoot')
				if table_foot:
					for row in table_foot.findAll('tr'):
						cells  = row.findAll('td')
						playerData = str(cells) #the indexes are not uniform across the database
						NCAA_fgpct = re.search(r'data-stat="fg_pct">(.*?)</td>', playerData).group(1)
						NCAA_games = re.search(r'data-stat="g">(.*?)</td>', playerData).group(1)
						NCAA_ppg = re.search(r'data-stat="pts_per_g">(.*?)</td>', playerData).group(1)
						NCAA_ft = re.search(r'data-stat="ft_pct">(.*?)</td>', playerData).group(1)
						NCAA_fgpg = re.search(r'data-stat="fg_per_g">(.*?)</td>', playerData).group(1)
						NCAA_fgapg = re.search(r'data-stat="fga_per_g">(.*?)</td>', playerData).group(1)
						NCAA_ftpg = re.search(r'data-stat="ft_per_g">(.*?)</td>', playerData).group(1)
						NCAA_ftapg = re.search(r'data-stat="fta_per_g">(.*?)</td>', playerData).group(1)
						if re.search(r'data-stat="fg3_per_g">(.*?)</td>', playerData) != None:
							NCAA__3ptpg = re.search(r'data-stat="fg3_per_g">(.*?)</td>', playerData).group(1)
						else:
							NCAA__3ptpg = None
						if re.search(r'data-stat="fg3a_per_g">(.*?)</td>', playerData) != None:
							NCAA__3ptapg = re.search(r'data-stat="fg3a_per_g">(.*?)</td>', playerData).group(1)
						else:
							NCAA__3ptapg = None
						if re.search(r'data-stat="fg3_pct">(.*?)</td>', playerData) != None:
							NCAA__3ptpct = re.search(r'data-stat="fg3_pct">(.*?)</td>', playerData).group(1)	
						else:
							NCAA__3ptpct = None
						if re.search(r'data-stat="efg_pct">(.*?)</td>', playerData) != None:
							NCAA_efgpct = re.search(r'data-stat="efg_pct">(.*?)</td>', playerData).group(1)
						else:
							NCAA_efgpct = None
						break

	player_entry = {'NBA_fg%':fgpct ,
	                'NBA_g_played': games,
	                'NBA_ppg': ppg,
	                'NBA_ft%': ft,
	                'NBA_fg_per_game': fgpg,
	                'NBA_fga_per_game': fgapg,
	                'NBA_ft_per_g': ftpg,
	                'NBA_fta_p_g': ftapg,
	                'NBA__3ptpg': _3ptpg,
	                'NBA__3ptapg': _3ptapg,
	                'NBA__3ptpct': _3ptpct,
	                'NBA_efgpct': efgpct,
					'NBA_PER': plyr_eff_rtg,
					'NBA_vorp': vorp,
					'NBA_bpm': bpm,
					'NBA_ws_per_48': ws_per_48,
					'NBA_ws': ws,
	                'NCAA_fgpct': NCAA_fgpct,
	                'NCAA_games': NCAA_games,
	                'NCAA_ppg' : NCAA_ppg,
	                'NCAA_ft': NCAA_ft,
	                'NCAA_fgpg': NCAA_fgpg,
	                'NCAA_fgapg': NCAA_fgapg,
	                'NCAA_ftpg': NCAA_ftpg,
	                'NCAA_ftapg': NCAA_ftapg,
	                'NCAA__3ptpg': NCAA__3ptpg,
	                'NCAA__3ptapg': NCAA__3ptapg,
	                'NCAA__3ptpct': NCAA__3ptpct,
	                'NCAA_efgpct': NCAA_efgpct
	                }
    
	return player_entry

def get_player_college_url(NBA_url):
	page_request = requests.get('http://www.basketball-reference.com' + str(url))
	soup = BeautifulSoup(page_request.text,"lxml")
	links = str(soup.findAll('li')) #regex time
	college_url = re.search(r'<a href="(.*?)">College Basketball at Sports-Reference.com</a>', links)
	if(college_url != None):
		return str(college_url.group(1))
	else:
		return None
	

######################################################################################
#MAIN 
players_general_info = player_basic_info() # call function that scrapes general info
print('General info/player url loaded...')
players_details_info_list = []
df = pd.DataFrame()	
bar = progressbar.ProgressBar(max_value=len(players_general_info))
for i,url in enumerate(players_general_info.url):
	try:
		player = player_info(url)
		df = df.append(player, ignore_index = True)
		print(df)
		bar.update(i)
		time.sleep(0.1)
	except Exception as e:
		print('Error caught:' + str(e))
print('Done!') #takes an unholy amount of time
df2 = pd.concat([players_general_info, df], axis =1)
df2 = df2.reindex(players_general_info.index)
df2.to_csv('player_data.csv', encoding='utf-8')


######################################################################################
