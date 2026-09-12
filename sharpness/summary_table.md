## Sharpness estimates

| Group     |   Top eigenvalue AdamW |   Top eigenvalue Muon |   Loss rise @σ=0.05 AdamW |   Loss rise @σ=0.05 Muon | Paired diff (A−M) ± SE   | A>M draws   |
|-----------|------------------------|-----------------------|---------------------------|--------------------------|--------------------------|-------------|
| hidden    |                 10.901 |                17.251 |                   0.0009  |                  0.00065 | +0.00025 ± 0.00034       | 6/10        |
| nonhidden |                  5.167 |                 3.243 |                   0.00154 |                  0.00084 | +0.00070 ± 0.00036       | 8/10        |
| all       |                 13.773 |                19.794 |                   0.00294 |                  0.00173 | +0.00121 ± 0.00061       | 8/10        |

## Generalization gap along the interpolation line

| Solution                   |   Train loss |   Test loss |    Gap |
|----------------------------|--------------|-------------|--------|
| AdamW (t=0)                |       0.0356 |      0.2757 | 0.2401 |
| Muon (t=1)                 |       0.0148 |      0.3167 | 0.3018 |
| Test-loss minimum (t=0.58) |       0.0119 |      0.2388 |        |
