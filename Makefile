.PHONY: train
train:
	PYTHONPATH=. python train_catboost.py

.PHONY: train_runker
train:
	PYTHONPATH=. python train_catboost_runker.py

.PHONY: run
run:
	PYTHONPATH=. python run_catboost.py
