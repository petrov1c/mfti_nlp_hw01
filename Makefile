.PHONY: train
train:
	PYTHONPATH=. python train_catboost.py

.PHONY: train_ranker
train_ranker:
	PYTHONPATH=. python train_catboost_ranker.py

.PHONY: run
run:
	PYTHONPATH=. python run_catboost.py
