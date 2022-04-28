from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from six import StringIO  
import pydot 
import pandas as pd

data_path = './player_data.csv'
data_csv = pd.read_csv(data_path)

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
ncaa_stats = ncaa_stats.drop(['index', 'Unnamed: 0'], axis='columns')

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
PER_X_train, PER_X_test, PER_y_train, PER_y_test = train_test_split(ncaa_stats, PER, test_size=0.25, random_state=42, stratify=PER)
vorp_X_train, vorp_X_test, vorp_y_train, vorp_y_test = train_test_split(ncaa_stats, vorp, test_size=0.25, random_state=42, stratify=vorp)
bpm_X_train, bpm_X_test, bpm_y_train, bpm_y_test = train_test_split(ncaa_stats, bpm, test_size=0.25, random_state=42, stratify=bpm)
ws48_X_train, ws48_X_test, ws48_y_train, ws48_y_test = train_test_split(ncaa_stats, ws_per_48, test_size=0.25, random_state=42, stratify=ws_per_48)
ws_X_train, ws_X_test, ws_y_train, ws_y_test = train_test_split(ncaa_stats, ws, test_size=0.25, random_state=42, stratify=ws)

# Create all decision trees and random forests
PER_dec_tree = tree.DecisionTreeClassifier()
PER_random_forest = RandomForestClassifier()

vorp_dec_tree = tree.DecisionTreeClassifier()
vorp_random_forest = RandomForestClassifier()

bpm_dec_tree = tree.DecisionTreeClassifier()
bpm_random_forest = RandomForestClassifier()

ws48_dec_tree = tree.DecisionTreeClassifier()
ws48_random_forest = RandomForestClassifier()

ws_dec_tree = tree.DecisionTreeClassifier()
ws_random_forest = RandomForestClassifier()

# Fit each decision tree and random forest
#bpm dec tree and ws48 rand forest
PER_dec_tree.fit(PER_X_train, PER_y_train)
PER_random_forest.fit(PER_X_train, PER_y_train)

vorp_dec_tree.fit(vorp_X_train, vorp_y_train)
vorp_random_forest.fit(vorp_X_train, vorp_y_train)

bpm_dec_tree.fit(bpm_X_train, bpm_y_train)
bpm_random_forest.fit(bpm_X_train, bpm_y_train)

ws48_dec_tree.fit(ws48_X_train, ws48_y_train)
ws48_random_forest.fit(ws48_X_train, ws48_y_train)

ws_dec_tree.fit(ws_X_train, ws_y_train)
ws_random_forest.fit(ws_X_train, ws_y_train)

# Print each decision tree and random forest accuracy
print('***DECISION TREE AND RANDOM FOREST SCORES***\n')
print('PER decision tree accuracy= ' + str(PER_dec_tree.score(PER_X_test, PER_y_test)))
print('PER random forest accuracy= ' + str(PER_random_forest.score(PER_X_test, PER_y_test)))
print('vorp decision tree accuracy= ' + str(vorp_dec_tree.score(vorp_X_test, vorp_y_test)))
print('vorp random forest accuracy= ' + str(vorp_random_forest.score(vorp_X_test, vorp_y_test)))
print('bpm decision tree accuracy= ' + str(bpm_dec_tree.score(bpm_X_test, bpm_y_test)))
print('bpm random forest accuracy= ' + str(bpm_random_forest.score(bpm_X_test, bpm_y_test)))
print('ws/48 decision tree accuracy= ' + str(ws48_dec_tree.score(ws48_X_test, ws48_y_test)))
print('ws/48 random forest accuracy= ' + str(ws48_random_forest.score(ws48_X_test, ws48_y_test)))
print('ws decision tree accuracy= ' + str(ws_dec_tree.score(ws_X_test, ws_y_test)))
print('ws random forest accuracy= ' + str(ws_random_forest.score(ws_X_test, ws_y_test)))

# Create decision tree graphics for each statistic
print('Generating decision tree graphics...')
dot_data = StringIO()
answers = ['IS NOT ALLSTAR', 'IS ALLSTAR']

export_graphviz(PER_dec_tree, out_file=dot_data, feature_names=ncaa_stats.columns, class_names=answers) 
dec_graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
dec_graph[0].write_pdf("PER_dec_tree.pdf")

export_graphviz(vorp_dec_tree, out_file=dot_data, feature_names=ncaa_stats.columns, class_names=answers) 
dec_graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
dec_graph[0].write_pdf("vorp_dec_tree.pdf")

export_graphviz(bpm_dec_tree, out_file=dot_data, feature_names=ncaa_stats.columns, class_names=answers) 
dec_graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
dec_graph[0].write_pdf("bpm_dec_tree.pdf")

export_graphviz(ws48_dec_tree, out_file=dot_data, feature_names=ncaa_stats.columns, class_names=answers) 
dec_graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
dec_graph[0].write_pdf("ws48_dec_tree.pdf")

export_graphviz(ws_dec_tree, out_file=dot_data, feature_names=ncaa_stats.columns, class_names=answers) 
dec_graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
dec_graph[0].write_pdf("ws_dec_tree.pdf")
print('Done!')