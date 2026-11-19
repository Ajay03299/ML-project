# Integration notes for the group review

These notes concern this measured contribution and the current starter code,
not a claim to have completed other members' analyses.

## For Pragun (reviewer)

- Review the 60-minute horizon on the actual hourly input: it is one step, not
  four. Requests for 15-minute targets are rejected.
- The old shared cache lacks a causal weather-fill mask. Every model in this
  run therefore uses the same 29 non-weather features. Your ablation script
  already skips groups absent from the baseline matrix; do not interpret a
  skipped weather/covariate group as evidence of zero predictive value.
- The PCA forest and plain forest have different validation-selected forest
  settings in the consolidated table. Their score difference alone is not an
  isolated causal test of PCA. Your dedicated controlled analysis should make
  that distinction.
- All canonical settings were selected on validation; test results did not
  trigger retuning. Finite boosting caps and a smaller tuning sample are
  limitations. Source/result manifests preserve the exact configuration.

## For Anirudh (evaluation/model-serving owner)

- Public evaluate_model/save-for-UI signatures are preserved. New metadata adds
  run/data/source identity, feature hashes, split/sample identity and model file
  hashes. Keep these when assembling the final team benchmark.
- Shared tune adds a backwards-compatible optional keyword out_dir. Existing
  calls retain their output behavior; supporting searches can now write isolated
  tables/parameters without overwriting another owner's results. A regression
  test protects the shared files in this mode.
- Model-size serialization is now disk-backed: a full-data forest can exceed
  3 GiB, so a BytesIO copy would create substantial memory pressure.
- Analysis E presently reads the saved metadata columns but does not validate
  data/file hashes as the UI does. Before the final run, check that it is using
  the same panel and frozen model artifacts; do not mix a later weather-enabled
  matrix with this non-weather cohort.
- If no wx_precip column is available, Analysis E correctly omits rain/dry
  segments. Explain that omission rather than describing unmeasured weather
  robustness.
- LightGBM here uses its default GBDT boosting type. Do not claim that GOSS was
  evaluated merely because it is mentioned in the starter description.
- The original shared throughput field times one large prediction batch. This
  contribution separately measures repeated single-row latency and batches;
  they answer different serving questions.

## For Ajay (data owner)

- The merged panel's bytes are unchanged. A future raw-based rebuild can retain
  weather only after causal filling is established; the new builder forward
  fills at most two hours and marks preprocessing version 2.
- The inherited 12-step zero-run rule means 12 hours on this input and marks
  whole runs retrospectively. It is archival quality screening, not a validated
  online fault detector; the report states that boundary.
- Sensor-holdout features currently derive correlations and sensor statistics
  from the complete training panel. Clarify whether historical observations of
  held-out IDs are permitted before calling the analysis completely cold-start
  sensor generalisation. This contribution tests future times on the same 60
  supervised sensor IDs.

The final team benchmark may change after other owners approve different
settings. Preserve this frozen snapshot as evidence of this run; re-benchmark
coherently after any changed feature/data/model configuration.

## Observed zero-run screening scope in this run

A direct before/after audit of the selected 60 sensors found 27 newly flagged
readings, all from sensor 10801 between 2022-08-07 13:00 and 2022-08-08 15:00.
This is entirely before the training boundary (2023-05-29 13:00). The filter
adds no flags within this run's validation or test periods. The rule is still
an archival screening assumption, not a confirmed sensor-fault label or an
online detection policy. The data-audit JSON records these counts and dates.

## Ablation and neighbour interpretation

The shared builder always retains slot, is_weekend and sensor_id for the
baselines and downstream grouping. Removing the calendar group therefore
removes its engineered encodings/holiday fields, not every calendar signal;
removing the sensor group removes summaries, not the sensor identifier. Label
those ablations precisely. The primary neighbour feature uses top-five
training-flow correlations; the map is geographic context, not proof that a
road-adjacency graph was trained. Distance neighbours are a separate variant.
