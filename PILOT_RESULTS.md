# Pilot analysis (3 scenarios, 300 questions)

Store sizes (cells / residuals): s1: 141/1424, s2: 153/1673, s3: 148/3097
Full-conversation tokens per scenario: s1: 6288, s2: 5732, s3: 6647

## Table A. Selection budget, cell-level chain recall, answer accuracy (95% bootstrap CIs)

| variant | mean cells | ctx tokens | ctx / full-conv | chain recall [CI] | answer acc [CI] (n) |
|---|---|---|---|---|---|
| prograph | 140.8 | 56941 | 9.1x | 1.000 [1.000, 1.000] | 0.897 [0.860, 0.930] (300) |
| prograph_t0.3 | 136.0 | 55811 | 8.9x | 0.997 [0.990, 1.000] | 0.885 [0.845, 0.921] (252) |
| prograph_t0.4 | 108.1 | 48391 | 7.7x | 0.923 [0.890, 0.950] | 0.890 [0.831, 0.941] (136) |
| percand | 84.6 | 41983 | 6.6x | 0.837 [0.793, 0.877] | 0.883 [0.847, 0.917] (300) |
| percand_t0.3 | 41.6 | 23928 | 3.8x | 0.503 [0.447, 0.560] | 1.000 [1.000, 1.000] (1) |
| percand_t0.4 | 16.3 | 10842 | 1.7x | 0.310 [0.257, 0.363] | 0.742 [0.581, 0.871] (31) |
| cosine_h0 | 5.0 | 4657 | 0.7x | 0.253 [0.203, 0.303] | 0.683 [0.633, 0.733] (300) |
| cosine_k10 | 10.0 | 7622 | 1.2x | 0.267 [0.217, 0.317] | — |
| cosine_k20 | 20.0 | 12636 | 2.0x | 0.323 [0.270, 0.373] | 0.877 [0.840, 0.913] (300) |
| cosine_k40 | 40.0 | 21842 | 3.5x | 0.430 [0.377, 0.483] | — |
| crossenc_k10 | 10.0 | 8050 | 1.3x | 0.390 [0.337, 0.443] | — |
| crossenc_k20 | 20.0 | 15134 | 2.4x | 0.533 [0.473, 0.587] | 0.890 [0.857, 0.923] (300) |
| crossenc_k40 | 40.0 | 26736 | 4.3x | 0.800 [0.753, 0.843] | — |
| cosine_mpc | 84.6 | 41920 | 6.6x | 0.820 [0.777, 0.863] | 0.920 [0.887, 0.947] (300) |
| crossenc_mpc | 84.6 | 46720 | 7.4x | 0.987 [0.973, 0.997] | 0.903 [0.870, 0.933] (300) |

## Table B. Paired comparisons (one-shot vs iterated at matched budget; positive = first is better)

| comparison | metric | mean diff [95% CI] | n |
|---|---|---|---|
| crossenc_mpc − percand | chain recall | +0.150 [+0.110, +0.190] | 300 |
| crossenc_mpc − percand | answer acc | +0.020 [-0.007, +0.050] | 300 |
| cosine_mpc − percand | chain recall | -0.017 [-0.033, -0.003] | 300 |
| cosine_mpc − percand | answer acc | +0.037 [+0.007, +0.067] | 300 |
| crossenc_k40 − percand_t0.3 | chain recall | +0.297 [+0.233, +0.357] | 300 |
| crossenc_k20 − cosine_h0 | chain recall | +0.280 [+0.230, +0.330] | 300 |
| crossenc_k20 − cosine_h0 | answer acc | +0.207 [+0.150, +0.263] | 300 |
| crossenc_k20 − prograph | chain recall | -0.467 [-0.527, -0.413] | 300 |
| crossenc_k20 − prograph | answer acc | -0.007 [-0.043, +0.033] | 300 |
| cosine_mpc − prograph | chain recall | -0.180 [-0.223, -0.137] | 300 |
| cosine_mpc − prograph | answer acc | +0.023 [-0.010, +0.057] | 300 |

## Table C. Expansion dynamics (mean cells after each step; chain recall)

| variant | step 0 | step 1 | step 2 | final | store |
|---|---|---|---|---|---|
| prograph | 5 / 0.25 | 103 / 1.00 | 141 / 1.00 | 141 / 1.00 | 147 |
| prograph_t0.3 | 5 / 0.25 | 103 / 1.00 | 136 / 1.00 | 136 / 1.00 | 147 |
| prograph_t0.4 | 5 / 0.25 | 95 / 0.92 | 108 / 0.92 | 108 / 0.92 | 147 |
| percand | 5 / 0.25 | 70 / 0.84 | 84 / 0.84 | 85 / 0.84 | 147 |
| percand_t0.3 | 5 / 0.25 | 38 / 0.50 | 42 / 0.50 | 42 / 0.50 | 147 |
| percand_t0.4 | 5 / 0.25 | 16 / 0.31 | 16 / 0.31 | 16 / 0.31 | 147 |

## Cost ledger (from usage.jsonl)

| tag | calls | prompt tokens | completion tokens |
|---|---|---|---|
| judge | 2088 | 257893 | 27090 |
| read:answer | 2103 | 65181001 | 143183 |
| write:identify | 61 | 27070 | 19323 |
| write:profile | 1175 | 1458079 | 1014226 |