import os

import pandas as pd
from catboost import Pool, CatBoostRanker
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()
ITERATIONS = int(os.environ['ITERATIONS'])
FULL = bool(int(os.environ['FULL']))
learning_rate = float(os.environ['learning_rate'])

TRAIN_COLUMNS = ['user', 'track', 'artist', 'genre', 'pop', 'duration']

tracks = pd.read_json("data/tracks.json", lines=True)
train_data = pd.read_csv("data/train.csv")

train_data = pd.merge(train_data, tracks, how='left', on='track')
train_data.genre = train_data.genre.map(lambda genres: genres[0])

train_split, val_split = train_test_split(train_data, test_size=0.1)
train_split.sort_index(inplace=True)
val_split.sort_index(inplace=True)

FULL = False
if FULL:
    X_train = train_data[TRAIN_COLUMNS]
    y_train = train_data['time']
else:
    X_train = train_split[TRAIN_COLUMNS]
    y_train = train_split['time']

X_val = val_split[TRAIN_COLUMNS]
y_val = val_split['time']

train_pool = Pool(data=X_train, label=y_train, cat_features=['user', 'track', 'artist', 'genre'], group_id=X_train.user)
val_pool = Pool(data=X_val, label=y_val, cat_features=['user', 'track', 'artist', 'genre'], group_id=X_val.user)

model = CatBoostRanker(
    iterations=ITERATIONS,
    custom_metric=['NDCG'],
    learning_rate=learning_rate,
    task_type="GPU",
    use_best_model=True,
    loss_function='YetiRank',
    metric_period=50,
)

best_model = model.fit(
    train_pool,
    eval_set=val_pool,
    verbose=2000,
    plot=False,
)

best_model.save_model('catboost_ranker.dump')
