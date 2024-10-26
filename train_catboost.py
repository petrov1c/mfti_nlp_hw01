import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split

VERSION = 6
SEED = 42
TRAIN_COLUMNS = ['user', 'track', 'artist', 'pop', 'duration']

tracks = pd.read_json("data/tracks.json", lines=True)
train_data = pd.read_csv("data/train.csv")

train_data = pd.merge(train_data, tracks, how='left', on='track')

# train_split, val_split = train_test_split(train_data, test_size=0.15, random_state=seed, shuffle=True, stratify=train_data['rate'])
train_split, val_split = train_test_split(train_data, test_size=0.1, random_state=SEED)

X_train = train_data[TRAIN_COLUMNS]
y_train = train_data['time']

X_val = pd.DataFrame(val_split[TRAIN_COLUMNS])
y_val = val_split['time']

train_pool = Pool(data=X_train, label=y_train, cat_features=['user', 'track', 'artist'])
val_pool = Pool(data=X_val, label=y_val, cat_features=['user', 'track', 'artist'])

model = CatBoostRegressor(
    iterations=150,
    learning_rate=0.2,
    random_seed=SEED,
    task_type="GPU",
    use_best_model=True,
    eval_metric='RMSE',
    save_snapshot=False,
#    max_ctr_complexity=2,
)

best_model = model.fit(
    train_pool,
    eval_set=val_pool,
    verbose=2000,
    plot=False,
)

best_model.save_model('catboost_model.dump')
