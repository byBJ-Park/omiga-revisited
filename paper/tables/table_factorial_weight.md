# OMIGA's weighted policy loss: tanh-Gaussian NLL actor vs squared-error actor (frozen critic, MuJoCo 2.0)

Same advantage weight exp(w (Q - V) / alpha) from OMIGA's critic in both columns; mean (sd) over 3 seeds, 100 fixed initial states.

| task | dataset | out of range | OMIGA actor (tanh-Gaussian NLL) | squared-error actor | difference |
|---|---|---|---|---|---|
| Hopper | expert | 28.4% | 923 (1348), surv 0.11 | 3341 (273), surv 0.94 | +2418 |
| Hopper | medium-expert | 19.2% | 92 (6), surv 0.00 | 1351 (1674), surv 0.35 | +1259 |
| Hopper | medium | 9.3% | 1137 (313), surv 0.14 | 1554 (48), surv 0.06 | +417 |
| Hopper | medium-replay | 0.3% | 590 (365), surv 0.05 | 1911 (222), surv 0.53 | +1321 |
| Ant | medium | 0.0% | 1425 (2), surv 1.00 | 1412 (2), surv 1.00 | -12 |

Per-seed returns:

- Hopper expert: OMIGA actor 108 / 2478 / 181; squared-error actor 3055 / 3368 / 3599
- Hopper medium-expert: OMIGA actor 96 / 94 / 85; squared-error actor 591 / 3270 / 191
- Hopper medium: OMIGA actor 1495 / 1004 / 912; squared-error actor 1609 / 1524 / 1530
- Hopper medium-replay: OMIGA actor 1006 / 324 / 439; squared-error actor 2013 / 2064 / 1656
- Ant medium: OMIGA actor 1423 / 1426 / 1426; squared-error actor 1411 / 1415 / 1412
