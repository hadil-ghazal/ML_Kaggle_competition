#No AI was used to generate this code. Authored by Hadil Ghazal on 2/25/26

#Initial Imports
import pandas as pd
import numpy as np

#V3 Haversine 
def haversine(lat1, lon1, lat2, lon2):
# I want to create a new datapoint focusing on distancs, 
# If a customer is close to the stadium it might be an influence in getting them more committed to a subscription and able to attend
# not too sure if this is a stretch but going to calculate in km by starting with the standard constant earths radius then calculating individual distance off that   
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


train = pd.read_csv('for_students/train.csv')
test = pd.read_csv('for_students/test.csv')
test.rename(columns={'ID': 'account.id'}, inplace=True)  # this is a temp patch. My join is failing without it. Need to fix in V2
accounts = pd.read_csv('for_students/account.csv',encoding='latin1')
subs = pd.read_csv('for_students/subscriptions.csv',encoding='latin1')
tickets = pd.read_csv('for_students/tickets_all.csv',encoding='latin1')
concerts = pd.read_csv('for_students/concerts.csv',encoding='latin1')
concerts14 = pd.read_csv('for_students/concerts_2014-15.csv',encoding='latin1')
zips = pd.read_csv('for_students/zipcodes.csv',encoding='latin1')

print("load complete")

#print("cip columns check:", zips.columns.tolist())

# step1 Merging train with the account join data

train_full = train.merge(accounts, on='account.id', how='left')
test_full  = test.merge(accounts, on='account.id', how='left')

train_full.head()

#final csv is failing, need to check here what the data is looking like
print("TRAIN COLUMNS:", train.columns.tolist())
print("ACCOUNTS COLUMNS:", accounts.columns.tolist())
print("MERGED TRAIN COLUMNS:", train_full.columns.tolist())

#import sys; sys.exit()

# 1.5 realized that one of the tables has IDs only not account ID so temp patch until rewritten better
if 'account.id' in test.columns:
    test_full = test.merge(accounts, on='account.id', how='left')
else:
    print("error id missing, Columns are:", test.columns.tolist())
#import sys; sys.exit()


# Step 2 - Clean numeric columns
train_full['amount.donated.lifetime'] = pd.to_numeric(train_full['amount.donated.lifetime'], errors='coerce')
train_full['no.donations.lifetime'] = pd.to_numeric(train_full['no.donations.lifetime'], errors='coerce')

test_full['amount.donated.lifetime'] = pd.to_numeric(test_full['amount.donated.lifetime'], errors='coerce')
test_full['no.donations.lifetime'] = pd.to_numeric(test_full['no.donations.lifetime'], errors='coerce')

# Step 3)  Derived Feature: Average contribution per donation
train_full['avg.contribution'] = train_full['amount.donated.lifetime'] / train_full['no.donations.lifetime']
test_full['avg.contribution']  = test_full['amount.donated.lifetime'] / test_full['no.donations.lifetime']

# Step 4) erived Feature: years since first contribution
train_full['first.year'] = pd.to_datetime(train_full['first.donated'], errors='coerce').dt.year
test_full['first.year']  = pd.to_datetime(test_full['first.donated'], errors='coerce').dt.year

train_full['years_since_first_contribution'] = 2026 - train_full['first.year']
test_full['years_since_first_contribution']  = 2026 - test_full['first.year']

# V2 adding one more derived feature , creating a column for distance to the stadium zip code which might influence purchase
STADIUM_ZIP = 15212 # I'm hardocding this zip code becasue setting the fixed reference point from the stadium
stadium_row = zips[zips['Zipcode'] == STADIUM_ZIP].iloc[0]
stad_lat, stad_lon = stadium_row['Lat'], stadium_row['Long']

    #Make the zip columns same type for merging
train_full['shipping.zip.code'] = train_full['shipping.zip.code'].astype(str)
test_full['shipping.zip.code'] = test_full['shipping.zip.code'].astype(str)
zips['Zipcode'] = zips['Zipcode'].astype(str)

train_full = train_full.merge(zips, left_on='shipping.zip.code', right_on='Zipcode', how='left')
test_full  = test_full.merge(zips, left_on='shipping.zip.code', right_on='Zipcode', how='left')

    #distance calc here
train_full['distance_to_stadium'] = haversine(
    train_full['Lat'], train_full['Long'], stad_lat, stad_lon
)
test_full['distance_to_stadium'] = haversine(
    test_full['Lat'], test_full['Long'], stad_lat, stad_lon
)

# Step 5) Aggregate subscriptions
subs_agg = subs.groupby('account.id')['season'].nunique().reset_index().rename(columns={'season':'num_subscription_seasons'})
tickets_agg = tickets.groupby('account.id')['season'].nunique().reset_index().rename(columns={'season':'num_ticket_seasons'})

# Step 6) Merge aggregates
train_full = train_full.merge(subs_agg, on='account.id', how='left')
train_full = train_full.merge(tickets_agg, on='account.id', how='left')

test_full = test_full.merge(subs_agg, on='account.id', how='left')
test_full = test_full.merge(tickets_agg, on='account.id', how='left')

# Step 7 -Fill NA  values
train_full = train_full.fillna(0)
test_full  = test_full.fillna(0)

# Step 8) Selecting numeric columns only
##feature_cols = ['total.contributed', 'no.of.donations', 'avg.contribution',
  ##              'years_since_first_contribution', 'num_subscription_seasons',
    ##            'num_ticket_seasons']
    ##V2 after change
feature_cols = [
    'amount.donated.lifetime', 
    'no.donations.lifetime',
    'avg.contribution',
    'years_since_first_contribution',
    'num_subscription_seasons',
    'num_ticket_seasons',
    'distance_to_stadium' 
]

X = train_full[feature_cols]
y = train_full['label']
X_test = test_full[feature_cols]

# Step 9 training a LightGBM model - I am keeping it simple during this version
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4
)

model.fit(X, y)
preds = model.predict_proba(X_test)[:,1]

#  step 10) generating the final submission file
#V2 I didn't realize kaggle needs exact column names, renamed from ID and predicted to match submission expectations
submission = pd.DataFrame({
    'ID': test['account.id'],   
    'Predicted': preds          
})

submission.to_csv('submission.csv', index=False)
print("submission.csv created successfully!")