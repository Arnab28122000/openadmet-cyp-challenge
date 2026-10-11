# Baseline arms — binary ECFP4 vs count-ECFP4 + RDKit2D

TDC official splits. `gain` is always in the direction of better.

| task            |     n | metric   |   binary |   count+rdkit2d |    gain | wins   |
|:----------------|------:|:---------|---------:|----------------:|--------:|:-------|
| ppbr            |  1614 | mae      |   9.8053 |          8.1026 |  1.7027 | Y      |
| solubility      |  9982 | mae      |   1.242  |          0.8036 |  0.4384 | Y      |
| caco2           |   910 | mae      |   0.4727 |          0.3164 |  0.1562 | Y      |
| lipophilicity   |  4200 | mae      |   0.6348 |          0.5259 |  0.1088 | Y      |
| ld50            |  7385 | mae      |   0.6822 |          0.6156 |  0.0666 | Y      |
| hia             |   578 | roc_auc  |   0.907  |          0.9728 |  0.0658 | Y      |
| cyp2d6_inhib    | 13130 | pr_auc   |   0.6412 |          0.6995 |  0.0583 | Y      |
| cyp2d6_sub      |   667 | pr_auc   |   0.6068 |          0.6606 |  0.0538 | Y      |
| half_life       |   667 | spearman |   0.2809 |          0.3341 |  0.0532 | Y      |
| cyp3a4_sub      |   670 | roc_auc  |   0.6362 |          0.6889 |  0.0527 | Y      |
| bioavailability |   640 | roc_auc  |   0.6245 |          0.6766 |  0.052  | Y      |
| bbb             |  2030 | roc_auc  |   0.8045 |          0.8449 |  0.0405 | Y      |
| ames            |  7278 | roc_auc  |   0.8184 |          0.8569 |  0.0384 | Y      |
| cyp2c9_inhib    | 12092 | pr_auc   |   0.7495 |          0.7835 |  0.0339 | Y      |
| dili            |   475 | roc_auc  |   0.8761 |          0.9004 |  0.0243 | Y      |
| cyp3a4_inhib    | 12328 | pr_auc   |   0.8544 |          0.877  |  0.0226 | Y      |
| cl_micro        |  1102 | spearman |   0.5462 |          0.5622 |  0.016  | Y      |
| herg            |   655 | roc_auc  |   0.7825 |          0.7898 |  0.0074 | Y      |
| cyp2c9_sub      |   669 | pr_auc   |   0.3415 |          0.3326 | -0.0089 | .      |
| pgp             |  1218 | roc_auc  |   0.9178 |          0.9085 | -0.0093 | .      |
| vdss            |  1130 | spearman |   0.5017 |          0.4623 | -0.0394 | .      |

**count+rdkit2d wins 18/21 endpoints.**
