# Connect4

This wrapper uses `gymnasium-connect-four==1.3.5` by instantiating `connect_four_gymnasium.ConnectFourEnv` directly. The default opponent is `BabySmarterPlayer` and `first_player` is `1`.

The observation is `MultiDiscreteSpace([3] * 42)`: empty cells are `0`, the current/agent perspective is `1`, and the opponent is `2`. The wrapper exposes `info["valid_actions"]`. Q-learning and SBQ(ours) use application-local masked agents that derive valid columns from the top row, sample only non-full columns, and randomly break ties among maximal valid Q values.

Q-learning, DQN, and SBQ(ours) configs share seed, 100k-step experiment schedule, evaluation cadence, evaluation episodes, evaluation seed, and visualization settings. SBQ(ours) uses the same SBQ update rule with an app-local Hamming neighborhood implementation, because the full board product space is too large for the generic `from_index` decoder.

Install into the `bcrl` env with:

```bash
conda run -n bcrl python -m pip install --no-deps gymnasium-connect-four==1.3.5
```
