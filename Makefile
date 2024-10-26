.PHONY: train
train:
	PYTHONPATH=. python train_catboost.py

.PHONY: run
run:
	PYTHONPATH=. python run_catboost.py
