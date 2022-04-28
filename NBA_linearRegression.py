import pandas as pd
from sklearn.linear_model import LinearRegression



data_path = './player_data.csv'
data_csv = pd.read_csv(data_path)

# We need to clean the data by removing all rows with 'None' values so fitting models is easier
data_csv = data_csv.drop(['NCAA_efgpct'], axis=1)
filtered_data_csv = data_csv.dropna()
filtered_data_csv = filtered_data_csv.reset_index()

# Get the y-values (advanced NBA statistics indicative of succcess)
PER = filtered_data_csv.NBA_PER
vorp = filtered_data_csv.NBA_vorp
bpm = filtered_data_csv.NBA_bpm
ws_per_48 = filtered_data_csv.NBA_ws_per_48
ws = filtered_data_csv.NBA_ws

# First, use college stats to see which statistic has the best correlation for NBA PER, vorp, bpm, ws/48, and ws using Linear Regression
# Get the x-values (NCAA shooting statistics)
ncaa_fgpct = filtered_data_csv.NCAA_fgpct.values.reshape(-1, 1)
ncaa_ppg = filtered_data_csv.NCAA_ppg.values.reshape(-1, 1)
ncaa_ft = filtered_data_csv.NCAA_ft.values.reshape(-1, 1)
ncaa_fgpg = filtered_data_csv.NCAA_fgpg.values.reshape(-1, 1)
ncaa_fgapg = filtered_data_csv.NCAA_fgapg.values.reshape(-1, 1)
ncaa_3ptpg = filtered_data_csv.NCAA__3ptpg.values.reshape(-1, 1)
ncaa_3ptapg = filtered_data_csv.NCAA__3ptapg.values.reshape(-1, 1)
ncaa_3ptpct = filtered_data_csv.NCAA__3ptpct.values.reshape(-1, 1)

# Fit all PER models
PER_fgpct_mdl = LinearRegression().fit(ncaa_fgpct, PER)
PER_ppg_mdl = LinearRegression().fit(ncaa_ppg, PER)
PER_ft_mdl = LinearRegression().fit(ncaa_ft, PER)
PER_fgpg_mdl = LinearRegression().fit(ncaa_fgpg, PER)
PER_fgapg_mdl = LinearRegression().fit(ncaa_fgapg, PER)
PER_threeptpg_mdl = LinearRegression().fit(ncaa_3ptpg, PER)
PER_threeptapg_mdl = LinearRegression().fit(ncaa_3ptapg, PER)
PER_threeptpct_mdl = LinearRegression().fit(ncaa_3ptpct, PER)

# Fit all vorp models
vorp_fgpct_mdl = LinearRegression().fit(ncaa_fgpct, vorp)
vorp_ppg_mdl = LinearRegression().fit(ncaa_ppg, vorp)
vorp_ft_mdl = LinearRegression().fit(ncaa_ft, vorp)
vorp_fgpg_mdl = LinearRegression().fit(ncaa_fgpg, vorp)
vorp_fgapg_mdl = LinearRegression().fit(ncaa_fgapg, vorp)
vorp_threeptpg_mdl = LinearRegression().fit(ncaa_3ptpg, vorp)
vorp_threeptapg_mdl = LinearRegression().fit(ncaa_3ptapg, vorp)
vorp_threeptpct_mdl = LinearRegression().fit(ncaa_3ptpct, vorp)

# Fit all bpm models
bpm_fgpct_mdl = LinearRegression().fit(ncaa_fgpct, bpm)
bpm_ppg_mdl = LinearRegression().fit(ncaa_ppg, bpm)
bpm_ft_mdl = LinearRegression().fit(ncaa_ft, bpm)
bpm_fgpg_mdl = LinearRegression().fit(ncaa_fgpg, bpm)
bpm_fgapg_mdl = LinearRegression().fit(ncaa_fgapg, bpm)
bpm_threeptpg_mdl = LinearRegression().fit(ncaa_3ptpg, bpm)
bpm_threeptapg_mdl = LinearRegression().fit(ncaa_3ptapg, bpm)
bpm_threeptpct_mdl = LinearRegression().fit(ncaa_3ptpct, bpm)

# Fit all ws/48 models
ws48_fgpct_mdl = LinearRegression().fit(ncaa_fgpct, ws_per_48)
ws48_ppg_mdl = LinearRegression().fit(ncaa_ppg, ws_per_48)
ws48_ft_mdl = LinearRegression().fit(ncaa_ft, ws_per_48)
ws48_fgpg_mdl = LinearRegression().fit(ncaa_fgpg, ws_per_48)
ws48_fgapg_mdl = LinearRegression().fit(ncaa_fgapg, ws_per_48)
ws48_threeptpg_mdl = LinearRegression().fit(ncaa_3ptpg, ws_per_48)
ws48_threeptapg_mdl = LinearRegression().fit(ncaa_3ptapg, ws_per_48)
ws48_threeptpct_mdl = LinearRegression().fit(ncaa_3ptpct, ws_per_48)

# Fit all ws models
ws_fgpct_mdl = LinearRegression().fit(ncaa_fgpct, ws)
ws_ppg_mdl = LinearRegression().fit(ncaa_ppg, ws)
ws_ft_mdl = LinearRegression().fit(ncaa_ft, ws)
ws_fgpg_mdl = LinearRegression().fit(ncaa_fgpg, ws)
ws_fgapg_mdl = LinearRegression().fit(ncaa_fgapg, ws)
ws_threeptpg_mdl = LinearRegression().fit(ncaa_3ptpg, ws)
ws_threeptapg_mdl = LinearRegression().fit(ncaa_3ptapg, ws)
ws_threeptpct_mdl = LinearRegression().fit(ncaa_3ptpct, ws)

# Print scores for the PER models
print('***LINEAR REGRESSION MODELS AND SCORES***\n')
print('***PER Models:***')
print('accuracy of fg%= ' + str(PER_fgpct_mdl.score(ncaa_fgpct, PER)*100) + '%')
print('accuracy of ppg= ' + str(PER_ppg_mdl.score(ncaa_ppg, PER)*100) + '%')
print('accuracy of ft%= ' + str(PER_ft_mdl.score(ncaa_ft, PER)*100) + '%')
print('accuracy of fgpg= ' + str(PER_fgpg_mdl.score(ncaa_fgpg, PER)*100) + '%')
print('accuracy of fgapg= ' + str(PER_fgapg_mdl.score(ncaa_fgapg, PER)*100) + '%')
print('accuracy of 3ptpg= ' + str(PER_threeptpg_mdl.score(ncaa_3ptpg, PER)*100) + '%')
print('accuracy of 3ptapg= ' + str(PER_threeptapg_mdl.score(ncaa_3ptapg, PER)*100) + '%')
print('accuracy of 3ptpct= ' + str(PER_threeptpct_mdl.score(ncaa_3ptpct, PER)*100) + '%')

# Print scores for the vorp models
print()
print('***VORP Models:***')
print('accuracy of fg%= ' + str(vorp_fgpct_mdl.score(ncaa_fgpct, vorp)*100) + '%')
print('accuracy of ppg= ' + str(vorp_ppg_mdl.score(ncaa_ppg, vorp)*100) + '%')
print('accuracy of ft%= ' + str(vorp_ft_mdl.score(ncaa_ft, vorp)*100) + '%')
print('accuracy of fgpg= ' + str(vorp_fgpg_mdl.score(ncaa_fgpg, vorp)*100) + '%')
print('accuracy of fgapg= ' + str(vorp_fgapg_mdl.score(ncaa_fgapg, vorp)*100) + '%')
print('accuracy of 3ptpg= ' + str(vorp_threeptpg_mdl.score(ncaa_3ptpg, vorp)*100) + '%')
print('accuracy of 3ptapg= ' + str(vorp_threeptapg_mdl.score(ncaa_3ptapg, vorp)*100) + '%')
print('accuracy of 3ptpct= ' + str(vorp_threeptpct_mdl.score(ncaa_3ptpct, vorp)*100) + '%')

# Print scores for the bpm models
print()
print('***BPM Models:***')
print('accuracy of fg%= ' + str(bpm_fgpct_mdl.score(ncaa_fgpct, bpm)*100) + '%')
print('accuracy of ppg= ' + str(bpm_ppg_mdl.score(ncaa_ppg, bpm)*100) + '%')
print('accuracy of ft%= ' + str(bpm_ft_mdl.score(ncaa_ft, bpm)*100) + '%')
print('accuracy of fgpg= ' + str(bpm_fgpg_mdl.score(ncaa_fgpg, bpm)*100) + '%')
print('accuracy of fgapg= ' + str(bpm_fgapg_mdl.score(ncaa_fgapg, bpm)*100) + '%')
print('accuracy of 3ptpg= ' + str(bpm_threeptpg_mdl.score(ncaa_3ptpg, bpm)*100) + '%')
print('accuracy of 3ptapg= ' + str(bpm_threeptapg_mdl.score(ncaa_3ptapg, bpm)*100) + '%')
print('accuracy of 3ptpct= ' + str(bpm_threeptpct_mdl.score(ncaa_3ptpct, bpm)*100) + '%')

# Print scores for the ws/48 models
print()
print('***WS/48 Models:***')
print('accuracy of fg%= ' + str(ws48_fgpct_mdl.score(ncaa_fgpct, ws_per_48)*100) + '%')
print('accuracy of ppg= ' + str(ws48_ppg_mdl.score(ncaa_ppg, ws_per_48)*100) + '%')
print('accuracy of ft%= ' + str(ws48_ft_mdl.score(ncaa_ft, ws_per_48)*100) + '%')
print('accuracy of fgpg= ' + str(ws48_fgpg_mdl.score(ncaa_fgpg, ws_per_48)*100) + '%')
print('accuracy of fgapg= ' + str(ws48_fgapg_mdl.score(ncaa_fgapg, ws_per_48)*100) + '%')
print('accuracy of 3ptpg= ' + str(ws48_threeptpg_mdl.score(ncaa_3ptpg, ws_per_48)*100) + '%')
print('accuracy of 3ptapg= ' + str(ws48_threeptapg_mdl.score(ncaa_3ptapg, ws_per_48)*100) + '%')
print('accuracy of 3ptpct= ' + str(ws48_threeptpct_mdl.score(ncaa_3ptpct, ws_per_48)*100) + '%')

# Print scores for the ws models
print()
print('***WS Models:***')
print('accuracy of fg%= ' + str(ws_fgpct_mdl.score(ncaa_fgpct, ws)*100) + '%')
print('accuracy of ppg= ' + str(ws_ppg_mdl.score(ncaa_ppg, ws)*100) + '%')
print('accuracy of ft%= ' + str(ws_ft_mdl.score(ncaa_ft, ws)*100) + '%')
print('accuracy of fgpg= ' + str(ws_fgpg_mdl.score(ncaa_fgpg, ws)*100) + '%')
print('accuracy of fgapg= ' + str(ws_fgapg_mdl.score(ncaa_fgapg, ws)*100) + '%')
print('accuracy of 3ptpg= ' + str(ws_threeptpg_mdl.score(ncaa_3ptpg, ws)*100) + '%')
print('accuracy of 3ptapg= ' + str(ws_threeptapg_mdl.score(ncaa_3ptapg, ws)*100) + '%')
print('accuracy of 3ptpct= ' + str(ws_threeptpct_mdl.score(ncaa_3ptpct, ws)*100) + '%')




