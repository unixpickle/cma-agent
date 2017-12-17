# CMA-ES for RL

This is a baseline for applying [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES) to simple continuous control tasks.

# Results

In some cases, CMA-ES is capable of finding pretty good policies. However, it is not very sample efficient for training neural networks.

Here is a table of mean rewards after training for a while. DDPG and CEM are included for comparison.

| Environment               | CMA-ES | DDPG   | CEM   |
|---------------------------|--------|--------|--------
| HalfCheetah               | 1321   | 3307   | -355  |
| Hopper                    | 178    | 699    | 813   |
| InvertedDoublePendulum    | 243    | 649    | 157   |
| InvertedPendulum          | 331    | 342    | 879   |
| Reacher                   | -12.5  | -14.4  | -12.1 |
| Swimmer                   | 22.3   | 38.5   | 7.02  |
| Walker2d                  | 136    | 1048   | 527   |

The `results/` directory includes the full training logs for CMA-ES.
