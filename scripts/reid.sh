# cache the attribute predictor's probabilities for test set
python -m src.reid.cache_probs --n_train_samples 500

# run a sample of reid top-k combinations (100 configs × baseline + model = 200 runs)
# each config is run as both baseline=True (random guess) and baseline=False (LM prediction)
# so the scatter plot can compare them
python scripts/run_reid_sample.py

# to run ALL 5760 combinations instead (slow), use batch_run with --execute_program:
# python -m src.reid.batch_run --start 0 --end 5759 --execute_program
# python -m src.reid.batch_run --start 0 --end 5759 --execute_program --baseline