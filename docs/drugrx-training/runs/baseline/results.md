# Baseline — ECFP4 + LightGBM

Generated 2026-10-04T17:29:29.

official_split is TDC's own scaffold split and is leaderboard-comparable; our_scaffold_splits is this repo's splitter and is an error-bar estimate only.

| task            |     n | metric   |   official |   ours_mean |   ours_std |
|:----------------|------:|:---------|-----------:|------------:|-----------:|
| caco2           |   910 | mae      |     0.3164 |      0.3486 |     0.0194 |
| bioavailability |   640 | roc_auc  |     0.6766 |      0.6946 |     0.0322 |
| lipophilicity   |  4200 | mae      |     0.5259 |      0.5584 |     0.0046 |
| solubility      |  9982 | mae      |     0.8036 |      0.9376 |     0.0162 |
| hia             |   578 | roc_auc  |     0.9728 |      0.9884 |     0.0081 |
| pgp             |  1218 | roc_auc  |     0.9085 |      0.9238 |     0.0251 |
| bbb             |  2030 | roc_auc  |     0.8449 |      0.9046 |     0.0183 |
| ppbr            |  1614 | mae      |     8.1026 |      9.1271 |     0.5958 |
| vdss            |  1130 | spearman |     0.4623 |      0.5299 |     0.0209 |
| cyp2c9_inhib    | 12092 | pr_auc   |     0.7835 |      0.7769 |     0.0141 |
| cyp2d6_inhib    | 13130 | pr_auc   |     0.6995 |      0.6982 |     0.0047 |
| cyp3a4_inhib    | 12328 | pr_auc   |     0.877  |      0.8332 |     0.0163 |
| cyp2c9_sub      |   669 | pr_auc   |     0.3326 |      0.2456 |     0.0256 |
| cyp2d6_sub      |   667 | pr_auc   |     0.6606 |      0.6696 |     0.0907 |
| cyp3a4_sub      |   670 | roc_auc  |     0.6889 |      0.6588 |     0.062  |
| half_life       |   667 | spearman |     0.3341 |      0.2986 |     0.0951 |
| cl_micro        |  1102 | spearman |     0.5622 |      0.4385 |     0.0421 |
| ld50            |  7385 | mae      |     0.6156 |      0.5644 |     0.0121 |
| herg            |   655 | roc_auc  |     0.7898 |      0.8544 |     0.0132 |
| ames            |  7278 | roc_auc  |     0.8569 |      0.8005 |     0.0122 |
| dili            |   475 | roc_auc  |     0.9004 |      0.8426 |     0.0225 |
