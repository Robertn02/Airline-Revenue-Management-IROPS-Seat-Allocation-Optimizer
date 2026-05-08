# Architecture

Detailed view of how the components fit together and the design decisions behind them.

## Pipeline

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│  Data           │───▶│  Risk estimator  │───▶│  LP allocator        │
│  generator      │    │                  │    │                      │
│                 │    │  GBT + Platt     │    │  HiGHS LP solver     │
│  • passengers   │    │  AUC 0.92        │    │  ~7ms median solve   │
│  • flights      │    │  Brier 0.058     │    │  Cohort-level        │
│  • inventory    │    │                  │    │                      │
│  • delays       │    │  Output:         │    │  Output:             │
│                 │    │  p_misconnect    │    │  per-pax assignment  │
│  Calibrated to  │    │  ∈ [0,1]         │    │  + audit log         │
│  BTS 2024       │    │  + confidence    │    │                      │
└─────────────────┘    └──────────────────┘    └──────────────────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────────┐
                                              │  Reason codes        │
                                              │  Constraint binding  │
                                              │  Cost decomposition  │
                                              └──────────────────────┘
```

## Why these choices

Three non-obvious decisions worth explaining.

### Why GBT not deep learning

Cohort sizes are small (10–60 passengers per scenario) and any realistic airline dataset would be small at the tail (~50–80 cascading disruption events per year per hub). Tree-based models handle small tabular data well, train quickly, and are interpretable. Calibration is recovered post-hoc via Platt scaling.

Deep learning would help if we had access to text (chat logs, agent notes) or sequences (historical interaction patterns), but the structured features here don't benefit from it.

### Why constrained LP not reinforcement learning

RL is architecturally appropriate for this problem — it's sequential and stateful. But three things make it premature:

1. **Data scarcity.** A major hub generates ~50–80 cascading disruption events per year. RL needs orders of magnitude more reward signal to converge on a stable policy. Even if you used scenario-level rewards rather than per-decision rewards, the sample efficiency isn't there.

2. **Auditability.** The LP produces a deterministic explanation for every decision: which constraints were binding, what the cost terms were, why this passenger got assigned to this flight in this cabin. RL policies don't expose this naturally, and that matters in a regulated environment.

3. **Sequencing.** Even if you wanted RL eventually, you'd need historical data to train an offline policy. The LP produces exactly that data — the structured audit log of decisions and outcomes that an offline RL policy could be trained on. The LP is a prerequisite for the learned system, not an alternative.

### Why (passenger, flight, cabin) variables not (passenger, flight)

An earlier version computed a single "best" cabin per (i,j) pair before solving, then used (passenger, flight) variables. This was a real bug: the LP couldn't find solutions that the manual baseline could find by opportunistically downgrading from F to Y when Y+ was full.

Expanding to (passenger, flight, cabin) variables tripled the variable count (172 → 750 for a 50-pax 5-flight scenario). Solve time stayed under 12 ms p95 because HiGHS is fast and the constraint matrix is sparse.

The fix is documented in tests/test_optimizer.py:test_lp_dominates_manual_in_aggregate, which would fail under the old formulation.

## Risk model

### Architecture

LightGBM classifier (200 estimators, 31 leaves, learning rate 0.05, min child samples 20). After training the base model, a second instance is fitted via `CalibratedClassifierCV` with `method="sigmoid"` (Platt scaling) and `cv=3`.

### Why calibration matters

The probability flows directly into the LP objective as a multiplier on the miss cost. Uncalibrated probabilities — even when AUC is high — distort this multiplier. A passenger with a calibrated probability of 0.30 has a misconnect cost of 0.30 × $X in expectation. If the raw GBT score were 0.50 for the same passenger, the LP would think the cost was 0.50 × $X and might overweight this passenger relative to others.

Niculescu-Mizil & Caruana 2005 is the foundational paper on this problem. Their finding: gradient boosted trees produce sigmoid-shaped score distributions that are well-suited to Platt scaling specifically (as opposed to random forests, which need isotonic regression).

### Feature importance via permutation, not split count

LightGBM's `feature_importances_` reports split count — the number of times each feature is used in a tree split. This is biased toward continuous features with many unique values, because they offer more split candidates.

Permutation importance shuffles each feature in the test set and measures the AUC degradation. This is a causal measure: how much does the model actually depend on this feature for its predictions?

Switching from split-count to permutation importance changed the reported top feature from `yield_usd` to `effective_buffer_min`, which matches the synthetic label model and matches what real domain knowledge says should matter.

This is not a cosmetic change. The earlier reporting was actively misleading — anyone reading the feature importance plot would have concluded that `yield_usd` was a primary driver of misconnect probability, which is not how the underlying system actually works.

## LP allocator

### Variables

For each passenger `i`, recovery flight `j`, and cabin `c ∈ {F, Y+, Y}`:
- `x_{i,j,c} ∈ [0, 1]` — passenger `i` assigned to flight `j` in cabin `c`
- `z_i ∈ [0, 1]` — passenger `i` unassigned (will misconnect with prob `p_i`)

### Objective

```
minimize  Σ_{i,j,c} x_{i,j,c} · (α·yield_dilution_{i,j} + β·spill_{i,j,c} + δ·harm_i)
        + Σ_i z_i · p_i · λ · (miss_cost_i + δ·harm_i)
```

The asymmetric probability weighting is the key insight:
- **Assignment costs** (`yield_dilution`, `spill`, `harm`) are paid deterministically. If we assign passenger i to flight j in cabin c, we pay these costs regardless of what would have happened to them otherwise.
- **Miss cost** is paid only with probability `p_i`. If we leave passenger i unassigned, they only actually misconnect with probability `p_i`; otherwise they make their original connection.

This is the correct expected-cost model. An earlier version weighted both by probability, which produced backward incentives.

### Constraints

```
Σ_{j,c} x_{i,j,c} + z_i = 1                  for each i  (each pax handled once)
Σ_i x_{i,j,c} ≤ open_seats[j,c]              for each (j,c)  (capacity)
x_{i,j,c} = 0  if MCT-infeasible              (hard MCT enforcement)
x_{i,j,c} = 0  if cabin not allowed           (loyalty floor for top tiers)
```

The loyalty floor prevents top-tier passengers (EXP, PLT) from being downgraded by more than one cabin. F → Y+ is allowed; F → Y is not, unless no alternative exists.

### Solver

SciPy's `linprog` with HiGHS backend. LP relaxation produces a fractional solution, then greedy integer rounding by sorted LP value. Empirically, the LP relaxation is integer-feasible on >99% of instances; the rounding step is a safety net.

## Manual triage baseline

The manual triage strategy models how a gate agent currently handles disruption recovery:

1. Sort passengers by tier (EXP > PLT > GLD > REG), then by yield descending.
2. For each passenger in priority order:
   - Try the original outbound flight first.
   - If full, try alternative recovery flights in earliest-departure order.
   - Within each flight, try same cabin first, then nearest-up, then nearest-down.
3. If all options exhausted, the passenger misconnects.

This is intentionally not optimized. It represents the realistic baseline the LP allocator competes against. The strategy is "rational at each step" but "globally suboptimal" — exactly the structure that creates the LP's improvement opportunity.

## Things that are simplified

- **Static inventory.** Real systems have inventory updating during the optimization window from normal booking activity. Would need rolling-horizon formulation.
- **No codeshare partners.** Only operator-controlled flights are considered.
- **No crew constraints.** Assumes the recovery flight will operate; doesn't model crew duty time.
- **No downstream cascading.** Reaccommodating a passenger to flight X doesn't model whether flight X has its own connecting passengers who'd be affected.
- **Single hub.** No multi-hub network optimization.

These are explicitly out of scope for what this exploration tries to demonstrate. They would all need to be addressed for a production system.

## Performance

Solve times measured on a M-series MacBook Air, single-threaded:

| Cohort size | Mean solve | P95 solve | Variables |
|---|---|---|---|
| 15 pax × 3 flights | 3.2 ms | 4.8 ms | 150 |
| 30 pax × 4 flights | 6.1 ms | 8.5 ms | 390 |
| 50 pax × 5 flights | 9.4 ms | 12.0 ms | 800 |

Far below the operational requirement (~200 ms) for live decision support during disruption windows.
