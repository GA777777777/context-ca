# Write-path propagation (12 injections, 89 affected cells in total)

| rule | rewrites / injection | tokens / injection (in/out) | affected cells missed | profile still stale | profile has new fact | residual still stale | collateral rewrites |
|---|---|---|---|---|---|---|---|
| R0_prograph | 8.4 | 14915 / 9786 | 54/89 | 51/89 | 33/89 | 62/89 | 66 |
| R1_comention | 41.1 | 56700 / 34578 | 0/89 | 2/89 | 87/89 | 39/89 | 404 |
| R2_ca_gate | 38.0 | 47424 / 26696 | 5/89 | 6/89 | 83/89 | 39/89 | 372 |
| R3_oracle | 12.9 | 21450 / 14169 | 0/89 | 1/89 | 87/89 | 38/89 | 66 |
| R4_oracle_evict | 12.9 | 39320 / 14396 | 0/89 | 1/89 | 88/89 | 15/89 | 66 |