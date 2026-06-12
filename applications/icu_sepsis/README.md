# ICU-Sepsis

This wrapper uses the external `icu-sepsis==2.0.1` package and the `Sepsis/ICU-Sepsis-v2` flattened-action environment.

The state is exact tabular mode: a single `state_id` factor with `MultiDiscreteSpace([716])`. The action space is `DiscreteActionSpace(25)`, corresponding to five IV fluid dose levels crossed with five vasopressor dose levels.

Default info includes the symbolic state id, admissible actions, SOFA score when provided by the package, terminal outcome, and action dose levels. Terminal outcome labels follow the package states: `713=death`, `714=survival`, and other terminal states are labeled `terminal`. The external environment has no true RGB render, so the wrapper visualizes the tracked trajectory as an RGB panel for GIF/PNG output.

All configs share seed, 200k-step experiment schedule, evaluation cadence, evaluation episodes, evaluation seed, visualization settings, and `gamma: 1.0`. SBQ(ours) uses generic Hamming-distance SBQ.

Install into the `bcrl` env with:

```bash
conda run -n bcrl python -m pip install --no-deps icu-sepsis==2.0.1
conda run -n bcrl python -m pip install pandas scikit-learn
```
