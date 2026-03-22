# Deep Q-Learning Formative 3 — ALE/Breakout-v5

**Environment:** ALE/Breakout-v5 (Atari Breakout)

**Framework:** Stable Baselines3 + Gymnasium

---

## Team Members

| Name | GitHub |
|---|---|
| Hirwa Armstrong Brian | [@Hirwabrian](https://github.com/Hirwabrian) |
| Ashimwe Carine | [@Carine-Ashimwe](https://github.com/Carine-Ashimwe) |
| Manzi Gustave | [@mgustavy](https://github.com/mgustavy) |
| Hammed Ayomide Agbaje | [@AgbajeCity](https://github.com/AgbajeCity) |

---

## Repository Structure

```
Deep-Q-Learning-Formative-3/
├── train.py                  # Training script (Task 1)
├── play.py                   # Evaluation/play script (Task 2)
├── select_best_model.py      # Utility: selects best model across all members
├── README.md
└── Models/
    ├── Armstrong/            # 10 model zips + hyperparameter logs
    ├── Ayomide/              # 10 model zips + hyperparameter logs + experiments.json
    ├── Carine/               # 10 model zips + hyperparameter logs
    └── Gustave/              # 10 model zips + hyperparameter logs
```

---

## How to Run

### Install Dependencies
```bash
pip install "gymnasium[atari,accept-rom-license]" stable-baselines3[extra] ale-py numpy imageio
```

### Train a Model
```bash
python train.py --config Models/Ayomide/experiments.json --exp-id 9 --env-id ALE/Breakout-v5 --steps 200000
```

### Play / Evaluate the Best Model
```bash
python play.py --model-path "Models/Carine/10_Optimized.zip" --env-id ALE/Breakout-v5 --episodes 5
```

### Find the Group Best Model
```bash
python select_best_model.py
```

---

## Gameplay Video

> 📹 Link to gameplay video of the best model (Carine — `10_Optimized`, mean reward: **24.4**) playing Breakout here:
>
> ---
>
> ## Hyperparameter Tuning Results
>
> Each group member ran **10 experiments** on `ALE/Breakout-v5` using a DQN agent, varying learning rate, gamma, batch size, and epsilon parameters. All experiments were trained for 200,000 steps.
>
> ---
>
> ### Armstrong — Hirwa Armstrong Brian
>
> | # | Model | Policy | lr | gamma | batch | ε_start | ε_end | ε_decay | Mean Reward | Noted Behavior |
> |---|---|---|---|---|---|---|---|---|---|---|
> | 1 | 01_Baseline | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 12.0 | Standard benchmark. Learned basic firing and initial ball tracking. |
> | 2 | 02_High_LR | MlpPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 15.6 | Faster but unstable. Higher reward but erratic movement. |
> | 3 | 03_Low_LR | CnnPolicy | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 8.6 | Too slow. Weights updated too slowly to react to fast ball speeds. |
> | 4 | 04_Low_Gamma | CnnPolicy | 5e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 10.4 | Short-sighted. Longest survival (14 steps) but lower reward. |
> | 5 | 05_High_Batch | CnnPolicy | 1e-4 | 0.80 | 32 | 1.0 | 0.05 | 0.10 | 21.0 | Top Performer. Stable gradients allowed agent to find high-scoring sweet spot. |
> | 6 | 06_Low_Batch | CnnPolicy | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | 7.6 | Erratic but survives. Long survival (40 steps) but poor scoring. |
> | 7 | 07_Fast_Decay | CnnPolicy | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | 11.6 | Premature Optimization. Stopped exploring too early. |
> | 8 | 08_High_Min_Eps | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | 15.6 | High Randomness. Stayed 50% random; surprisingly good score. |
> | 9 | 09_MLP_Comparison | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.40 | 0.10 | 6.0 | Poor Spatial Logic. Scored 50% less than CNN baseline. |
> | 10 | **10_Optimized** ⭐ | CnnPolicy | 2e-4 | 0.95 | 64 | 1.0 | 0.02 | 0.15 | **21.0** | The "Pro" Model. Combined high batch and optimized decay to reach maximum score. |
>
> **Best:** `10_Optimized` — Mean Reward **21.0**
>
> ---
>
> ### Ayomide — Hammed Ayomide Agbaje
>
> | # | Model | Policy | lr | gamma | batch | ε_start | ε_end | ε_decay | Mean Reward | Noted Behavior |
> |---|---|---|---|---|---|---|---|---|---|---|
> | 1 | A01_Baseline_CNN | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 11.4 | Standard baseline. Agent learned basic paddle movement and began tracking the ball. Stable but unoptimized. |
> | 2 | A02_High_LR | CnnPolicy | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 16.8 | Higher LR accelerated learning but introduced instability. Erratic paddle movement at times. |
> | 3 | A03_Very_Low_LR | CnnPolicy | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 6.2 | Very slow LR severely hampered the agent. Weights updated too slowly for fast ball dynamics. |
> | 4 | A04_High_Gamma | CnnPolicy | 1e-4 | 0.995 | 32 | 1.0 | 0.05 | 0.10 | 13.0 | Near-future-maximizing discount. Agent placed high value on long-term rewards but converged slowly. |
> | 5 | A05_Low_Gamma | CnnPolicy | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | 17.4 | Short-sighted but effective. Lower gamma pushed agent to exploit immediate rewards aggressively. |
> | 6 | A06_Large_Batch | CnnPolicy | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.10 | 19.6 | Larger batch provided more stable gradient estimates. Consistent top-tier performance. |
> | 7 | A07_Small_Batch | CnnPolicy | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | 8.8 | Very small batch led to noisy updates. High variance prevented effective learning. |
> | 8 | A08_Slow_Epsilon_Decay | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 14.2 | Slower exploration decay kept the agent exploring longer. Good balance of exploration/exploitation. |
> | 9 | A09_MLP_Policy | MlpPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 5.6 | MLP lacked spatial feature extraction for image-based tasks. Scored ~51% less than CNN baseline. |
> | 10 | **A10_Optimized** ⭐ | CnnPolicy | 2.5e-4 | 0.97 | 64 | 1.0 | 0.01 | 0.20 | **22.8** | Best config. Balanced LR with moderate gamma and large batch. Very low ε_end ensured near-greedy exploitation at convergence. |
>
> **Best:** `A10_Optimized` — Mean Reward **22.8**
>
> ---
>
> ### Carine — Ashimwe Carine
>
> | # | Model | Policy | lr | gamma | batch | ε_start | ε_end | ε_decay | Mean Reward | Noted Behavior |
> |---|---|---|---|---|---|---|---|---|---|---|
> | 1 | 01_Baseline | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 11.8 | Baseline performance; mid-table reference. |
> | 2 | 02_High_LR | CnnPolicy | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 18.4 | Strong score; high LR helped in this run. |
> | 3 | 03_Low_LR | CnnPolicy | 5e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 10.8 | Underperformed; learning likely too slow. |
> | 4 | 04_Low_Gamma | CnnPolicy | 1e-4 | 0.80 | 32 | 1.0 | 0.05 | 0.10 | 23.0 | Excellent score; short-horizon strategy worked well. |
> | 5 | 05_High_Batch | CnnPolicy | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | 18.0 | High-performing and stable candidate. |
> | 6 | 06_Low_Batch | CnnPolicy | 1e-4 | 0.99 | 8 | 1.0 | 0.05 | 0.10 | 13.0 | Decent result; noisy updates still viable. |
> | 7 | 07_Fast_Decay | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | 11.8 | Early exploration cutoff gave baseline-level result. |
> | 8 | 08_High_Min_Eps | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.50 | 0.10 | 16.0 | Good score with sustained exploration. |
> | 9 | 09_MLP_Comparison | MlpPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 7.6 | Lowest score; CNN remains better for image input. |
> | 10 | **10_Optimized** ⭐ | CnnPolicy | 2e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | **24.4** | Best overall. Optimized LR, larger batch, and low ε_end produced the highest reward in the group. |
>
> **Best:** `10_Optimized` — Mean Reward **24.4** 🏆 *(Group Best)*
>
> ---
>
> ### Gustave — Manzi Gustave
>
> | # | Model | Policy | lr | gamma | batch | ε_start | ε_end | ε_decay | Avg Score | Noted Behavior |
> |---|---|---|---|---|---|---|---|---|---|---|
> | 1 | G01_Baseline | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 0.0 | Baseline with default CNN parameters. |
> | 2 | G02_Higher_LR | CnnPolicy | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 0.0 | Higher learning rate; agent did not converge within training budget. |
> | 3 | G03_Very_Low_LR | CnnPolicy | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 0.0 | Very low LR; updates too slow for agent to learn meaningful policy. |
> | 4 | G04_High_Gamma | CnnPolicy | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.10 | 0.0 | High gamma; long-horizon discounting did not help within limited steps. |
> | 5 | G05_Low_Gamma | CnnPolicy | 1e-4 | 0.85 | 32 | 1.0 | 0.05 | 0.10 | 0.67 | Low gamma showed slight improvement; short-horizon rewards easier to learn. |
> | 6 | **G06_Large_Batch** ⭐ | CnnPolicy | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.10 | **2.0** | Larger batch produced the best result; more stable gradient estimates. |
> | 7 | G07_Slow_Explore | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 0.0 | Slower epsilon decay; agent kept exploring but did not score. |
> | 8 | G08_Low_Eps_End | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | 1.0 | Lower epsilon end; marginally better exploitation at end of training. |
> | 9 | G09_MLP_Policy | MlpPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 0.0 | MLP policy; spatial features not captured — worst performer as expected. |
> | 10 | G10_Best_Combo | CnnPolicy | 3e-4 | 0.97 | 48 | 1.0 | 0.02 | 0.20 | 0.0 | Combined optimizations; model may need more training steps to converge. |
>
> **Best:** `G06_Large_Batch` — Avg Score **2.0**
>
> ---
>
> ## Group Best Model Summary
>
> | Member | Best Model | Mean Reward |
> |---|---|---|
> | Armstrong | 10_Optimized | 21.0 |
> | Ayomide | A10_Optimized | 22.8 |
> | **Carine** | **10_Optimized** | **24.4 🏆** |
> | Gustave | G06_Large_Batch | 2.0 |
>
> **Group winner: Carine's `10_Optimized`** with a mean reward of **24.4**.
>
> To run the group's best model:
> ```bash
> python play.py --model-path "Models/Carine/10_Optimized.zip" --env-id ALE/Breakout-v5 --episodes 5
> ```
>
> ---
>
> ## Key Insights from Hyperparameter Tuning
>
> **What improved performance:**
> - **Larger batch sizes** (64–128) consistently produced more stable gradient updates and higher rewards across all members.
> - - **Moderate learning rates** (2e-4 to 5e-4) outperformed both very high and very low values. Too low (1e-5) caused underfitting; too high introduced instability.
>   - - **Lower epsilon_end** (0.01–0.02) ensured the agent fully exploited its learned policy at convergence rather than continuing to explore randomly.
>     - - **CNN policy** was significantly better than MLP for all members — on average ~50% higher reward — confirming that convolutional layers are essential for extracting spatial features from Atari frames.
>      
>       - **What harmed performance:**
>       - - **Very small batch sizes** (8–16) introduced noisy gradient updates and erratic agent behaviour.
>         - - **Very low learning rates** (1e-5) caused the agent to learn too slowly within the 200,000-step budget.
>           - - **MLP policy** was consistently the worst performer across all members due to its inability to process raw pixel inputs spatially.
>             - - **Fast epsilon decay** caused the agent to stop exploring too early before finding good strategies.
>              
>               - ---
>
> ## Individual Contributions
>
> | Member | Contribution |
> |---|---|
> | **Armstrong** | Set up the shared Colab notebook, designed the `train.py` script, ran 10 experiments, coordinated the group |
> | **Carine** | Created the GitHub repository, ran 10 experiments, pushed `play.py`, coordinated file structure |
> | **Ayomide** | Ran 10 experiments, created `experiments.json`, wrote `select_best_model.py` utility, documented hyperparameter observations |
> | **Gustave** | Ran 10 experiments, pushed model files |
