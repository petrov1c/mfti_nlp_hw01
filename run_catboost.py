import os
import pandas as pd
from catboost import Pool, CatBoostRegressor
from dotenv import load_dotenv

load_dotenv()

VERSION = os.environ['VERSION']

TRAIN_COLUMNS = ['user', 'track', 'artist', 'genre', 'pop', 'duration']

tracks = pd.read_json("data/tracks.json", lines=True)
test_data = pd.read_csv("data/test.csv")
test_data = pd.merge(test_data, tracks, how='left', on='track')
test_data.genre = test_data.genre.map(lambda genres: genres[0])

test_pool = Pool(data=test_data[TRAIN_COLUMNS], cat_features=['user', 'track', 'artist', 'genre'])

model = CatBoostRegressor()
model.load_model('catboost_model.dump')

y_test = model.predict(test_pool)

test_data['score'] = y_test
test_data[['user','track','score']].to_csv(f'submit/submit_catboost_{VERSION}.csv', index=False)

test_data.score = test_data.score.clip(0, 1)
test_data[['user','track','score']].to_csv(f'submit/submit_catboost_clip_{VERSION}.csv', index=False)
