# GTA VI launch-day Twitch forecast

Entry for make.fun `gta6-twitch-ccu-forecast-01`: forecast how many people watch
Grand Theft Auto VI live on Twitch in each hour of its release day
(2026-11-19 UTC), as five quantiles an hour, from everything recorded up to
the end of the day's first hour.

## How it forecasts

Launch days share one shape, driven by the rolling midnight unlock and the
time zones waking up in turn: a dip in the small hours, then a climb to a
peak in the late afternoon/evening UTC. `main.py` learns that shape from the
past launches shipped in the case file (`comparables`), each reduced to its
release day's 24 hourly counts:

- For every comparable with a complete release day, work in logs with
  `x = ln(hour-1 viewers)` and `y_h = ln(viewers in hour h / hour-1 viewers)`.
- Fit one straight line per hour, `y_h ~ a_h + b_h * x`. The slope is capped
  at ±0.5 and the line is re-centred, so a launch far bigger than anything in
  the comparables still gets a sane curve (its peak stays near hour 1 instead
  of collapsing).
- The spread for hour `h` is the residual scatter of the big launches only
  (hour-1 viewers ≥ 50,000), because small categories are sampled sparsely
  and their hour-1 prints are noisy. Quantiles are the median times
  `exp(z * spread)` with the standard normal's `z` at 0.05 / 0.25 / 0.5 /
  0.75 / 0.95.
- With too few usable comparables it falls back to persistence: every hour at
  hour 1's print with a widening random-walk spread. Any surprise in a case
  file falls back the same way instead of crashing.

Python 3.13, standard library only, no network, deterministic.

## Local practice results

Scored against the four practice launches (SullyGnome hourly counts, local
evaluation only), mean pinball loss on the log scale:

| forecaster | p00 | p01 | p02 | p03 | mean |
|---|---|---|---|---|---|
| house baseline | 0.744 | 0.287 | 0.215 | 0.303 | 0.387 |
| this entry | 0.144 | 0.160 | 0.127 | 0.238 | 0.167 |

90% coverage 0.92, typical miss 32% (baseline: 0.78 and 90%). Runtime about
0.2 s per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `*.out` per case.
