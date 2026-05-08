# Data Calibration Sources

Every distribution and parameter in the synthetic data generator is calibrated to a public source. This document enumerates them with citations.

## Why synthetic data

Real airline operational data is proprietary. Every published academic paper on airline disruption recovery either uses synthetic data calibrated to public sources or works under an NDA that prevents code release. The Reroute generator follows the academic convention: explicit assumptions, fully reproducible, not pretending to be real airline data.

## Delay distributions

**Source:** [BTS On-Time Performance data, 2024](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp)

- ~22% of US flights arrive >15 min late (`bts_late_fraction = 0.22`)
- Mean delay among late flights: ~50 min (`bts_delay_mean_min = 50`)
- Conditional delay distribution approximated as lognormal with sigma 0.85 (`bts_delay_lognormal_sigma = 0.85`) — fits the heavy right tail of real delay distributions

## Loyalty tier distribution

**Source:** Industry reporting on AAdvantage and similar US carrier programs.

```python
tier_distribution = {
    "EXP": 0.04,   # Executive Platinum / Diamond / equivalent
    "PLT": 0.09,   # Platinum / Gold (top 10–15%)
    "GLD": 0.18,   # Gold / Silver
    "REG": 0.69,   # Most passengers have no status
}
```

## Cabin distribution

**Source:** Typical narrow-body domestic configurations (Boeing 737-800 / Airbus A321 with three-cabin layout).

```python
cabin_distribution = {
    "F": 0.08,    # First / Business
    "Y+": 0.18,   # Premium economy / Main Cabin Extra
    "Y": 0.74,    # Main cabin
}
```

## Yield distributions

**Source:** ATPCO published fare structures, normalized to typical mean fares for US domestic.

```python
yield_lognormal = {
    "F":  (7.45, 0.32),   # mean ~$1900
    "Y+": (6.60, 0.35),   # mean ~$800
    "Y":  (5.95, 0.40),   # mean ~$420
}
```

These are lognormal `(mu, sigma)` parameters for the natural log of yield in USD.

## Tier yield premium multipliers

**Source:** Reasoning, not direct citation. Top-tier passengers tend to fly higher fare classes more often (corporate travel, business need).

```python
tier_yield_multiplier = {"EXP": 1.45, "PLT": 1.25, "GLD": 1.10, "REG": 1.00}
```

## MCT (Minimum Connection Time)

**Source:** [American Airlines published MCT for DFW](https://www.aa.com/i18n/travel-info/connections/connections.jsp), typical for major US hubs.

```python
mct_domestic_min = 35  # minutes
ssr_handling_min = 60  # extended for SSR / unaccompanied minor
```

## Load factors

**Source:** [BTS-published industry load factor data, 2024](https://www.transtats.bts.gov/) — mean ~83% for US domestic. Peak banks at major hubs run higher.

In the generator, recovery-flight load factors are sampled from `Uniform(0.92, 0.98)` because we're modeling **peak banks during a disruption window** specifically — this is what creates the seat scarcity that makes the allocation problem interesting. Off-peak banks would have more spare capacity and the optimization opportunity would be smaller.

## Service recovery cost

**Source:** Industry rules of thumb, ~$200–400 per misconnect handling.

```python
miss_fixed_cost_usd = 250.0  # admin + voucher baseline
```

## Synthetic label generator (for risk model training)

The misconnect labels in `synthesize_misconnect_labels` are generated from a logistic model on operational features only:

```
logit = base + adjustments + noise

base depends on effective_buffer (= scheduled_buffer - actual_delay):
    < 0           : 4.5  (~99% prob — already missed)
    [0, 35)       : 2.2  (~90% — below MCT)
    [35, 60)      : 0.2  (~55% — tight)
    [60, 90)      : -1.6 (~17% — comfortable)
    >= 90         : -3.2 (~4%  — safe)

adjustments:
    +1.0 if has_ssr            (extra handling time required)
    +1.3 if unaccompanied_minor (significantly more handling)
    -0.5 if EXP tier            (fast-track services help marginally)
    -0.25 if PLT tier

noise = N(0, 0.55) per passenger
```

**Important:** Yield, cabin, and "tier-as-yield-proxy" are deliberately EXCLUDED from the label model. This is the fix for a label artifact in an earlier version where high-yield passengers spuriously appeared correlated with misconnects through the cabin selection pipeline. The current label generator depends only on operational features that would actually drive a real misconnect.

The fix is verified by `tests/test_risk_model.py:test_label_artifact_fixed`, which asserts that `effective_buffer_min` is the top permutation-importance feature and that `yield_usd` importance is below 10% of buffer's importance.

## What is NOT calibrated to public data

A few things in the system are reasoned through rather than cited:

- **Cost coefficients in the optimizer objective** (`alpha_yield = 1.0`, `beta_spill = 0.85`, etc.) are reasonable defaults but a real airline would tune them based on internal economics.
- **Yield dilution percentage** (12% of fare for same-day rebook) is an industry rule of thumb, not a published figure.
- **Spill cost percentage** (25% per cabin step down) is also a rule of thumb.
- **The displacement cost function** is currently zero — a real airline would model this based on bumped passenger costs.

These are flagged in code comments as places that would need real airline data to validate.
