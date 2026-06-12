# Taxi-v4 application

This application wraps Gymnasium `Taxi-v4` without changing the official reward
or transition semantics.

## State

Gymnasium exposes the Taxi state as one integer in `Discrete(500)`. The wrapper
decodes it into the official symbolic factors:

```python
MultiDiscreteSpace([5, 5, 5, 4])
```

The factors are `[taxi_row, taxi_col, passenger_location, destination]`.

## Actions And Rewards

The six discrete actions are south, north, east, west, pickup, and dropoff.
Rewards are passed through exactly from Gymnasium: normal steps are `-1`,
illegal pickup/dropoff is `-10`, and successful dropoff is `+20`.

`success` is true only when Gymnasium terminates on a positive dropoff reward.

## SBQ

Taxi has a clean factored symbolic state, but passenger and destination factors
represent task phase. The SBQ config therefore uses Hamming radius `1`, moderate
kernel strength, and moderate kernel updates.

## Structural SBQ

`config_sbq_structural.yaml` uses `taxi_structural_sbq`, an application-local
SBQ subclass that preserves the same state representation but changes
broadcasting semantics:

- distance is Manhattan distance over `[taxi_row, taxi_col]`;
- states with different `passenger_location` or `destination` have infinite
  distance and are excluded from neighborhoods;
- neighborhoods keep passenger and destination fixed and include taxi positions
  within Manhattan radius `2`.

This shares value locally across taxi positions while avoiding broadcasting
across different pickup/dropoff phases or goals.

## Visualization

`TaxiEnv.render(mode="rgb_array")` draws a compact RGB grid with colored
landmarks, walls, the passenger, and the taxi. When visualization is enabled,
the existing trainer can save real GIF rollouts and final PNG frames.

The configs use `visualization.max_steps: 201` so the trainer can append and
save a final frame even when Taxi reaches its 200-step time limit.
