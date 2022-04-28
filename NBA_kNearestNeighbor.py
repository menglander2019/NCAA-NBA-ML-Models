import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_path = './player_data.csv'
data_csv = pd.read_csv(data_path)

scaler = StandardScaler()

# Values used to separate allstars versus non-allstars for each advanced statistic
PER_CUTOFF = 18 # 20% greater than the average NBA player PER score (15.0)
VORP_CUTOFF = 20
BPM_CUTOFF = 2.0
WS48_CUTOFF = 0.13
WS_CUTOFF = 40

# We need to clean the data by removing all rows with 'None' values so fitting models is easier
data_csv = data_csv.drop(['NCAA_efgpct'], axis=1)
filtered_data_csv = data_csv.dropna()
filtered_data_csv = filtered_data_csv.reset_index()

# NCAA stats will be used as the x-values for the logistic regression model
ncaa_stats = filtered_data_csv.drop(['url', 'name', 'active_from', 'active_to', 'position', 
                            'college', 'height', 'weight', 'birth_date', 'NBA_fg%',
                            'NBA_g_played', 'NBA_ppg', 'NBA_ft%', 'NBA_fg_per_game',
                            'NBA_fga_per_game', 'NBA_ft_per_g', 'NBA_fta_p_g', 'NBA__3ptpg',
                            'NBA__3ptapg', 'NBA__3ptpct', 'NBA_efgpct', 'NBA_PER',
                            'NBA_vorp', 'NBA_bpm', 'NBA_ws_per_48', 'NBA_ws'],
                            axis='columns')

# Scale the data first
scaled_ncaa_stats = scaler.fit_transform(ncaa_stats)

# Now add additional columns that categorize player's based on advanced statistics values
filtered_data_csv['PER_is_allstar'] = pd.cut(filtered_data_csv['NBA_PER'], bins=[-100, PER_CUTOFF, float('Inf')],labels=['NO', 'YES'])
filtered_data_csv['vorp_is_allstar'] = pd.cut(filtered_data_csv['NBA_vorp'], bins=[-100, VORP_CUTOFF, float('Inf')],labels=['NO', 'YES'])
filtered_data_csv['bpm_is_allstar'] = pd.cut(filtered_data_csv['NBA_bpm'], bins=[-100, BPM_CUTOFF, float('Inf')],labels=['NO', 'YES'])
filtered_data_csv['ws48_is_allstar'] = pd.cut(filtered_data_csv['NBA_ws_per_48'], bins=[-100, WS48_CUTOFF, float('Inf')],labels=['NO', 'YES'])
filtered_data_csv['ws_is_allstar'] = pd.cut(filtered_data_csv['NBA_ws'], bins=[-100, WS_CUTOFF, float('Inf')],labels=['NO', 'YES'])

# Get the y-values (advanced NBA statistics indicative of succcess)
PER = filtered_data_csv.PER_is_allstar
vorp = filtered_data_csv.vorp_is_allstar
bpm = filtered_data_csv.bpm_is_allstar
ws_per_48 = filtered_data_csv.ws48_is_allstar
ws = filtered_data_csv.ws_is_allstar

# Separate the NCAA stats into training and testing data for each advanced statistic
PER_X_train, PER_X_test, PER_y_train, PER_y_test = train_test_split(scaled_ncaa_stats, PER, test_size=0.25, random_state=42, stratify=PER)
vorp_X_train, vorp_X_test, vorp_y_train, vorp_y_test = train_test_split(scaled_ncaa_stats, vorp, test_size=0.25, random_state=42, stratify=vorp)
bpm_X_train, bpm_X_test, bpm_y_train, bpm_y_test = train_test_split(scaled_ncaa_stats, bpm, test_size=0.25, random_state=42, stratify=bpm)
ws48_X_train, ws48_X_test, ws48_y_train, ws48_y_test = train_test_split(scaled_ncaa_stats, ws_per_48, test_size=0.25, random_state=42, stratify=ws_per_48)
ws_X_train, ws_X_test, ws_y_train, ws_y_test = train_test_split(scaled_ncaa_stats, ws, test_size=0.25, random_state=42, stratify=ws)

# Value of k determined here by repeated testing
k = 4
PER_knn = KNeighborsClassifier(k)
vorp_knn = KNeighborsClassifier(k)
bpm_knn = KNeighborsClassifier(k)
ws48_knn = KNeighborsClassifier(k)
ws_knn = KNeighborsClassifier(k)

# Fit each KNN classifier using training data
PER_knn.fit(PER_X_train, PER_y_train)
vorp_knn.fit(vorp_X_train, vorp_y_train)
bpm_knn.fit(bpm_X_train, bpm_y_train)
ws48_knn.fit(ws48_X_train, ws48_y_train)
ws_knn.fit(ws_X_train, ws_y_train)

# Print out each KNN classifier's accuracy score
print('***KNN CLASSIFIER SCORES***\n')
print('PER testing score= ' + str(PER_knn.score(PER_X_test, PER_y_test)))
print('vorp testing score= ' + str(vorp_knn.score(vorp_X_test, vorp_y_test)))
print('bpm testing score= ' + str(bpm_knn.score(bpm_X_test, bpm_y_test)))
print('ws/48 testing score= ' + str(ws48_knn.score(ws48_X_test, ws48_y_test)))
print('ws testing score= ' + str(ws_knn.score(ws_X_test, ws_y_test)))
