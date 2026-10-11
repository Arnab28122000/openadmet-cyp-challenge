# Baseline — ECFP4 + LightGBM

Generated 2026-10-04T17:33:34.

official_split is TDC's own scaffold split and is leaderboard-comparable; our_scaffold_splits is this repo's splitter and is an error-bar estimate only.

| task            |     n | metric   |   official |   ours_mean |   ours_std |
|:----------------|------:|:---------|-----------:|------------:|-----------:|
| caco2           |   910 | mae      |     0.4727 |      0.4118 |     0.0289 |
| bioavailability |   640 | roc_auc  |     0.6245 |      0.6208 |     0.029  |
| lipophilicity   |  4200 | mae      |     0.6348 |      0.6847 |     0.0164 |
| solubility      |  9982 | mae      |     1.242  |      1.4007 |     0.0241 |
| hia             |   578 | roc_auc  |     0.907  |      0.9266 |     0.0333 |
| pgp             |  1218 | roc_auc  |     0.9178 |      0.8724 |     0.0229 |
| bbb             |  2030 | roc_auc  |     0.8045 |      0.8584 |     0.0416 |
| ppbr            |  1614 | mae      |     9.8053 |     11.0045 |     0.3287 |
| vdss            |  1130 | spearman |     0.5017 |      0.4985 |     0.0219 |
| cyp2c9_inhib    | 12092 | pr_auc   |     0.7495 |      0.7325 |     0.0059 |
| cyp2d6_inhib    | 13130 | pr_auc   |     0.6412 |      0.64   |     0.0096 |
| cyp3a4_inhib    | 12328 | pr_auc   |     0.8544 |      0.8132 |     0.0227 |
| cyp2c9_sub      |   669 | pr_auc   |     0.3415 |      0.2413 |     0.071  |
| cyp2d6_sub      |   667 | pr_auc   |     0.6068 |      0.5539 |     0.0876 |
| cyp3a4_sub      |   670 | roc_auc  |     0.6362 |      0.6388 |     0.028  |
| half_life       |   667 | spearman |     0.2809 |      0.1685 |     0.0415 |
| cl_micro        |  1102 | spearman |     0.5462 |      0.3943 |     0.01   |
| ld50            |  7385 | mae      |     0.6822 |      0.5707 |     0.0108 |
| herg            |   655 | roc_auc  |     0.7825 |      0.8419 |     0.0499 |
| ames            |  7278 | roc_auc  |     0.8184 |      0.7737 |     0.0093 |
| dili            |   475 | roc_auc  |     0.8761 |      0.8357 |     0.0772 |
