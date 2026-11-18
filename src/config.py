"""One place for the settings every script shares, so all four of us run the same experiment.

Ajay changes SOURCE to "madrid" (and commits it) after the data gate. Nobody else edits this
file without telling the team. Command-line flags (--source, --n-sensors, ...) override it for a run.
"""
SOURCE = "synthetic"         # "synthetic" until the data gate passes, then "madrid" (or "pems04")
N_SENSORS = 60               # evenly spaced subset while developing; 300 for the final run if time allows
HORIZON = 4                  # steps ahead: 4 x 15 min = 60 minutes (main); 1 = 15 minutes (second horizon)
MAX_TRAIN_ROWS = 200_000     # training rows sampled for tuning and the benchmark
TUNE_ROWS = 60_000           # smaller sample used inside hyper-parameter searches
HOLDOUT_FRAC = 0.15          # share of sensors never seen in training (sensor-holdout test)
RANDOM_STATE = 42
