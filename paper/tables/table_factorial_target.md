# Regression target x loss (behaviour cloning, MuJoCo 2.0)

Mean (sd) over 3 training seeds of the mean return on 100 fixed initial states; surv = share of episodes reaching 1000 steps (seed mean). Out of range = share of dataset action components with |a| > 1.

| task | dataset | out of range | atanh target, NLL | atanh target, MSE | action target, NLL | action target, MSE |
|---|---|---|---|---|---|---|
| Hopper | expert | 28.4% | 439 (493), surv 0.05 | 98 (6), surv 0.00 | 3160 (666), surv 0.88 | 3470 (115), surv 0.98 |
| Hopper | medium-expert | 19.2% | 146 (50), surv 0.01 | 117 (37), surv 0.00 | 2981 (752), surv 0.78 | 3112 (708), surv 0.76 |
| Hopper | medium-replay | 0.3% | 165 (15), surv 0.00 | 1358 (584), surv 0.25 | 344 (165), surv 0.03 | 439 (122), surv 0.00 |
| HalfCheetah | expert | 42.9% | 2633 (177) | 2373 (941) | 4701 (144) | 4774 (48) |
| HalfCheetah | medium-expert | 31.7% | 2037 (125) | 2013 (315) | 4272 (260) | 4376 (188) |
| Ant | medium | 0.0% | 1418 (3), surv 1.00 | 1414 (2), surv 1.00 | 1416 (10), surv 1.00 | 1416 (5), surv 1.00 |

Per-seed returns:

- Hopper expert: atanh target, NLL 218 / 94 / 1004; atanh target, MSE 104 / 99 / 92; action target, NLL 3538 / 2391 / 3552; action target, MSE 3467 / 3357 / 3587
- Hopper medium-expert: atanh target, NLL 112 / 203 / 124; atanh target, MSE 159 / 98 / 92; action target, NLL 3569 / 2133 / 3239; action target, MSE 2300 / 3435 / 3600
- Hopper medium-replay: atanh target, NLL 182 / 157 / 156; atanh target, MSE 951 / 2026 / 1095; action target, NLL 245 / 534 / 253; action target, MSE 579 / 354 / 384
- HalfCheetah expert: atanh target, NLL 2789 / 2441 / 2669; atanh target, MSE 1587 / 2116 / 3416; action target, NLL 4757 / 4537 / 4808; action target, MSE 4770 / 4728 / 4824
- HalfCheetah medium-expert: atanh target, NLL 2031 / 1916 / 2165; atanh target, MSE 1711 / 1988 / 2339; action target, NLL 3996 / 4511 / 4311; action target, MSE 4159 / 4477 / 4491
- Ant medium: atanh target, NLL 1415 / 1418 / 1421; atanh target, MSE 1413 / 1416 / 1412; action target, NLL 1423 / 1405 / 1420; action target, MSE 1419 / 1410 / 1418
