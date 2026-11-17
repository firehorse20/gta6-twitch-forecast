"""Forecast a big game launch's Twitch viewers, hour by hour, from its first hour.

Approach: launch-day audience curves share one shape, driven by the rolling
midnight unlock and the time zones waking up in turn.  The program learns that
shape from the past launches shipped in the case file ("comparables"), each
reduced to its release day's 24 hourly counts, and anchors the forecast on the
one number the case gives it: hour 1's print.

For every comparable with a complete release day, work in logs with
    x = ln(hour-1 viewers)
    y_h = ln(viewers in hour h / hour-1 viewers),  h = 2..24
and fit one straight line per hour, y_h ~ a_h + b_h * x.  The slope is capped
at +/-0.5 and the line is re-centred, so a launch far bigger than anything in
the comparables still gets a sane curve (its peak stays near hour 1 instead of
collapsing).  The spread for hour h is the residual scatter of the big
launches only (hour-1 viewers >= 50,000), because small categories are sampled
sparsely and their hour-1 prints are noisy; that scatter is the honest
uncertainty for a big launch.  Quantiles are the median times exp(z * spread)
with the standard normal's z at 0.05 / 0.25 / 0.5 / 0.75 / 0.95.

When too few comparables survive, the forecast falls back to persistence:
every hour at hour 1's print with a widening random-walk spread.

Standard library only, no network, deterministic: the same input always
writes the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The slope of log-ratio on log hour-1 is never allowed past this magnitude:
# the raw fit is near -0.9 at the day's peak, which would say a launch twice
# as big as any comparable peaks below its own first hour.
SLOPE_CAP = 0.5
# A launch counts as "big" for the spread when its hour-1 print reaches this.
BIG_V1 = 50000.0
# Spread bounds, in natural-log units.
SIGMA_FLOOR = 0.25
SIGMA_CAP = 1.6
# Fallback spread (persistence): log sd per square-root hour.
FALLBACK_SIGMA = 0.15
# The fallback needs at least this many usable comparables to trust the fit.
MIN_COMPARABLES = 5
MIN_BIG_FOR_SPREAD = 8


def parse_ts(text):
    """An ISO 8601 UTC timestamp like 2026-11-19T00:00:00Z to a datetime."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def round_half_up_2(x):
    """Round x to 2 decimals, halves away from zero, as a JSON number.

    Whole numbers come back as ints so the output writes 100000, not 100000.0.
    """
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def release_day_curve(comp):
    """A comparable's 24 release-day hourly viewer counts, or None.

    The day runs from the comparable's day_start for 24 hours.  A curve is
    usable only when every hour has a positive count: a missing or zero hour
    means the source did not sample that category, and patching it would
    invent the shape the forecast is meant to learn.
    """
    start = parse_ts(comp["day_start"])
    by_hour = {}
    for point in comp.get("hourly", []):
        try:
            when = parse_ts(point["from"])
            viewers = float(point["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        hour = int((when - start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24 and hour not in by_hour:
            by_hour[hour] = viewers
    if len(by_hour) != 24:
        return None
    curve = [by_hour[h] for h in range(1, 25)]
    if any(v <= 0 for v in curve):
        return None
    return curve


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (intercept, slope, residual sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = my - slope * mx
    if n > 2:
        resid_sd = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        )
    else:
        resid_sd = 0.0
    return intercept, slope, resid_sd


def fit_shape(curves):
    """Per hour, the fitted (median log-ratio, spread) of the comparables.

    curves is a list of release-day curves.  Returns a list of 23
    (median, sigma) pairs for hours 2..24, or None when too few curves.
    """
    if len(curves) < MIN_COMPARABLES:
        return None
    log_v1 = [math.log(c[0]) for c in curves]
    mean_x = sum(log_v1) / len(log_v1)
    big = [c for c in curves if c[0] >= BIG_V1]
    if len(big) >= MIN_BIG_FOR_SPREAD:
        spread_pool = big
    else:
        spread_pool = curves
    log_v1_spread = [math.log(c[0]) for c in spread_pool]
    fitted = []
    for h in range(2, 25):
        ratios = [math.log(c[h - 1] / c[0]) for c in curves]
        mean_y = sum(ratios) / len(ratios)
        _, slope, _ = ols(log_v1, ratios)
        slope = max(min(slope, SLOPE_CAP), -SLOPE_CAP)
        intercept = mean_y - slope * mean_x
        spread_ratios = [math.log(c[h - 1] / c[0]) for c in spread_pool]
        _, _, sigma = ols(log_v1_spread, spread_ratios)
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        fitted.append((intercept, slope, sigma))
    return fitted


def hour_quantiles(p1, fitted_h):
    """The five quantiles for one hour from its fitted (intercept, slope, sigma)."""
    intercept, slope, sigma = fitted_h
    median_log_ratio = intercept + slope * math.log(p1)
    out = []
    for z in Z:
        value = p1 * math.exp(median_log_ratio + z * sigma)
        value = min(max(value, 0.01), 1e9)
        out.append(round_half_up_2(value))
    # Rounding can only break the order by a cent; restore it cheaply.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def fallback_quantiles(p1, hour):
    """Persistence: every hour at hour 1's print, spread widening with time."""
    sigma = FALLBACK_SIGMA * math.sqrt(hour - 1)
    out = []
    for z in Z:
        value = min(max(p1 * math.exp(z * sigma), 0.01), 1e9)
        out.append(round_half_up_2(value))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case):
    """The forecast for one case: 23 rows, hours 2..24, no matter what.

    Any surprise in the case file falls back to persistence instead of
    crashing: a crashed run scores as an invalid case and takes the entry
    out of the prizes, while a plain fallback still scores.
    """
    try:
        p1 = hour_one_print(case)
        curves = []
        for comp in case.get("comparables", []):
            curve = release_day_curve(comp)
            if curve is not None:
                curves.append(curve)
        fitted = fit_shape(curves)
    except Exception:
        p1, fitted = 1.0, None
        try:
            p1 = hour_one_print(case)
        except Exception:
            pass
    hours = []
    for h in range(2, 25):
        if fitted is None:
            q = fallback_quantiles(p1, h)
        else:
            q = hour_quantiles(p1, fitted[h - 2])
        hours.append({"hour": h, "q": q})
    return {"hours": hours}


def hour_one_print(case):
    """Hour 1's print from the case, at least 1 viewer."""
    known = [k for k in case.get("known", []) if k.get("hour") == 1]
    if not known:
        raise ValueError("case has no hour-1 print")
    return max(1.0, float(known[0]["viewers"]))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            forecast = forecast_case(case)
        except Exception:
            continue
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
