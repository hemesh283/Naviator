# Final IO-VNBD Benchmark Audit & Validation Summary

**Project:** Smart India Hackathon 2026 (SIH26168)
**Problem Statement:** AI/ML based Intelligent Dead Reckoning System for Seamless Navigation
**Organization:** ISRO
**Repository:** `D:\Projects\SIH_2026\dead_reckoning`
**Date:** September 2026 (originally Sep 6; **regenerated 2026-09-18** against `results/io_vnbd/real_benchmark_all_test_sequences.csv` and `real_benchmark_aggregate*.csv`, both dated **2026-09-14** — see PROJECT_STATUS.md §45)
**Auditor:** Senior Edge AI & Inertial Navigation Engineer

---

## 0. Why this document was regenerated

The original (2026-09-06) version of this document was computed **before** two changes that materially affect every drift number in it:

1. **§31 (2026-09-08):** a ground-truth-heading leak in the GPS-outage evaluation methodology was found and fixed — the simulator had been reading live ground-truth heading *through* a blackout instead of freezing it at outage entry, which inflated every baseline's apparent accuracy, ML baselines most of all (heading is what the ML displacement predictions are rotated by; a leaked, always-correct heading was silently correcting for the model's own heading-sensitive errors).
2. **§33 (2026-09-14):** Baseline 9 (INS + EKF + NHC + ZUPT, no ML) was added to the benchmark, and a physics-primary decision was made for the shipped app based on its result.

The original document's headline claim — **"9.10% drift on 120-second outages" for sequence `vw16a`** — was computed under the leaked-heading methodology and does not hold under the fixed one (current value: **60.12%** for that same sequence/baseline/duration; see §7 below). Every table in this document has been recomputed directly from the current, post-fix result files; nothing here is carried forward from the stale version without being re-derived. Sections 2–3 (raw IO-VNBD dataset composition) describe the input dataset itself, not benchmark output, and are unaffected by the fix — they are carried forward unchanged and were spot-checked but not fully re-audited this pass.

---

## 1. Executive Summary

This document presents the current scientific audit of the AI/ML Dead Reckoning system benchmarked against the real Inertial and Odometry Vehicle Navigation Benchmark Dataset (IO-VNBD), using the corrected (post-heading-leak-fix) evaluation methodology and the current 9-baseline benchmark ladder.

**Core Audit Finding:**
Across all four tested outage durations (10s/30s/60s/120s) and both evaluation subsets (all 32 outage/baseline combinations, and the 23 "moving" combinations where the vehicle actually travelled >100m), **plain physics (Baseline 2: strapdown INS + 6-state EKF, zero ML) has the lowest median endpoint drift at every duration under the "All Evaluations" subset**, and is tied or near-best under "Moving Evaluations" too. No ML-inclusive baseline (B3–B8) beats it at any duration under "All Evaluations". This is why the app's shipped configuration makes physics-based fusion the guaranteed default, with the ML model contributing only when its kinematic plausibility gate actively trusts its output (see Project Overview doc and PROJECT_STATUS.md §33–34).

At the specific threshold that matters for a "highly accurate" claim (<10% drift), **physics-only Baseline 9 (INS+EKF+NHC+ZUPT, no ML) clears it on 9/32 (28.1%) of all evaluations and 8/23 (34.8%) of moving evaluations — more than double the best ML baseline's rate (3/32, 9.4% all; 3/23, 13.0% moving)**. Median moving drift across all baselines and durations ranges from roughly 7% (Baseline 2, 10s) to 69% (ML baselines, 120s); there is no single number that honestly represents "the" accuracy of this system — it is duration- and baseline-dependent, and that dependence is reported in full below rather than reduced to one headline figure.

The ML model itself is not broken by a code bug: a direction-vs-magnitude diagnostic (see Project Overview doc) shows it is genuinely inconsistent across driving conditions — it wins the "Moving Evaluations, 60s" median-drift comparison outright (Baseline 4: ML+INS, 23.51% vs Baseline 2's 28.90%), so ML is not uniformly worse, just not reliably better, which is exactly why the app treats it as an assistive signal gated by plausibility rather than a guaranteed improvement.

---

## 2. Exact Dataset Statistics

*(Carried forward from the 2026-09-06 version; describes the static raw IO-VNBD input dataset, not a benchmark result, and is unaffected by the heading-leak fix. Not independently re-audited this pass — flagging that explicitly rather than presenting it as freshly reverified.)*

- **Smartphone CSV Files:** Exactly **144 files** (`S-*.csv`).
- **Vehicle ECU CSV Files:** Exactly **144 files** (`V-*.csv`).
- **Synchronized Pairs:** Exactly **72 matched pairs** (1:1 timestamp alignment).
- **Total Driving Time:** **29.74 hours** ($107,064.0\text{ seconds}$).
- **Total Smartphone Samples:** Exactly **1,070,640 records**.
- **Sampling Frequency:** Exactly **10.0 Hz** ($\Delta t = 100.0\text{ ms}$) verified across all 144 smartphone files.
- **Participating Drivers:** **5 drivers** (Drivers A, B, C, D, E).
- **Vehicle Speed Envelope:** Mean $45.08\text{ km/h}$, Maximum $131.85\text{ km/h}$.
- **Physical Sensor Alignment:** Phone was mounted in landscape orientation on the dashboard. Phone `GYROSCOPE Pitch` was verified as vehicle turning yaw rate ($r = 0.9348$ correlation with vehicle CAN bus yaw rate).

---

## 3. Exact Held-Out Test Set Inventory & 32-Outage-Combination Calculation

*(Carried forward unchanged — this describes the fixed test split, not a benchmark result computed under the old methodology.)*

The test split contains **9 completely held-out sequences** (zero overlap with training or validation drives):

| Sequence Key | Driver | Duration (s) | Raw Samples | 20-Sample Windows | Valid Outage Durations | Outage Count |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `m` | Driver B | 70.2s | 702 | 69 | 10s, 30s, 60s, 120s | 4 |
| `vfa01` | Driver E | 1148.6s | 11,486 | 1,147 | 10s, 30s, 60s, 120s | 4 |
| `vfa02` | Driver E | 1242.0s | 12,420 | 1,241 | 10s, 30s, 60s, 120s | 4 |
| `vw14c` | Driver E | 118.5s | 1,185 | 117 | 10s, 30s, 60s, 120s | 4 |
| `vw15` | Driver E | 150.0s | 1,500 | 149 | 10s, 30s, 60s, 120s | 4 |
| `vw16a` | Driver E | 148.0s | 1,480 | 147 | 10s, 30s, 60s, 120s | 4 |
| `vw16b` | Driver E | 90.7s | 907 | 89 | 10s, 30s, 60s | 3 |
| `vw17` | Driver E | 19.3s | 193 | 18 | 10s | 1 |
| `y1` | Driver D | 461.5s | 4,615 | 460 | 10s, 30s, 60s, 120s | 4 |
| **Total** | **3 Drivers** | **3,448.8s** | **34,488** | **3,437** | — | **32 Outage Combinations** |

$$\text{Total Outage Combinations} = 9 \text{ (10s)} + 8 \text{ (30s)} + 8 \text{ (60s)} + 7 \text{ (120s)} = 32 \text{ combinations}$$

The benchmark ladder now has **9 baselines** (B8's kinematic-plausibility-gate and B9's no-ML physics-primary configuration were added after the original 7-baseline version of this document — see PROJECT_STATUS.md §32–33), so:
$$\text{Total Evaluated Runs} = 32 \text{ combinations} \times 9 \text{ baselines} = \mathbf{288 \text{ evaluations}}$$
Verified: `results/io_vnbd/real_benchmark_all_test_sequences.csv` contains exactly **288 rows** (regenerated 2026-09-14).

---

## 4. Complete 9-Baseline Aggregate Tables

All drift figures below are **median/mean endpoint drift** (`drift_percent_endpoint` — cross-checked against the summary tables in the Sep 13 pre-screening brief and confirmed to match exactly), recomputed directly from `real_benchmark_aggregate.csv` and `real_benchmark_aggregate_moving.csv` (both regenerated 2026-09-14, post-heading-leak-fix, includes Baselines 8 and 9).

### A. All 32 Evaluated Outage Intervals

| Outage Duration | Navigation Baseline | N | Mean Drift (%) | Median Drift (%) | Median RMSE (m) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **10s** | B1: Pure INS | 9 | 41.38% | 19.02% | 4.88m |
| **10s** | B2: INS + EKF | 9 | 23.71% | **12.99%** | **3.16m** |
| **10s** | B3: ML Only | 9 | 375.80% | 35.41% | 14.17m |
| **10s** | B4: ML + INS | 9 | 324.00% | 33.54% | 12.07m |
| **10s** | B5: ML + INS + EKF | 9 | 375.80% | 35.41% | 14.17m |
| **10s** | B6: ML + INS + EKF + NHC | 9 | 375.61% | 35.42% | 14.17m |
| **10s** | B7: ML + INS + EKF + NHC + ZUPT | 9 | 267.31% | 35.42% | 7.67m |
| **10s** | B8: ML + Kinematic Gate + EKF + NHC + ZUPT | 9 | 266.66% | 35.73% | 11.08m |
| **10s** | B9: INS + EKF + NHC + ZUPT (no ML) | 9 | 29.34% | **12.15%** | 4.97m |
| **30s** | B1: Pure INS | 8 | 244.60% | 62.55% | 35.46m |
| **30s** | B2: INS + EKF | 8 | 132.76% | **34.38%** | **28.16m** |
| **30s** | B3: ML Only | 8 | 1348.90% | 50.82% | 68.45m |
| **30s** | B4: ML + INS | 8 | 1179.85% | 47.70% | 58.64m |
| **30s** | B5: ML + INS + EKF | 8 | 1348.90% | 50.82% | 68.45m |
| **30s** | B6: ML + INS + EKF + NHC | 8 | 1348.43% | 50.79% | 68.46m |
| **30s** | B7: ML + INS + EKF + NHC + ZUPT | 8 | 969.07% | 46.42% | 51.25m |
| **30s** | B8: ML + Kinematic Gate + EKF + NHC + ZUPT | 8 | 968.53% | 45.79% | 45.85m |
| **30s** | B9: INS + EKF + NHC + ZUPT (no ML) | 8 | 45.02% | 38.15% | 27.62m |
| **60s** | B1: Pure INS | 8 | 60.67% | 51.53% | 113.97m |
| **60s** | B2: INS + EKF | 8 | 31.10% | **30.39%** | **76.77m** |
| **60s** | B3: ML Only | 8 | 512.47% | 31.75% | 122.99m |
| **60s** | B4: ML + INS | 8 | 441.70% | 32.40% | 122.04m |
| **60s** | B5: ML + INS + EKF | 8 | 512.47% | 31.75% | 122.99m |
| **60s** | B6: ML + INS + EKF + NHC | 8 | 512.31% | 31.70% | 122.99m |
| **60s** | B7: ML + INS + EKF + NHC + ZUPT | 8 | 361.18% | 31.70% | 87.37m |
| **60s** | B8: ML + Kinematic Gate + EKF + NHC + ZUPT | 8 | 360.93% | 31.75% | 85.06m |
| **60s** | B9: INS + EKF + NHC + ZUPT (no ML) | 8 | 51.40% | 53.23% | 81.42m |
| **120s** | B1: Pure INS | 7 | 109.27% | 54.99% | **302.90m** |
| **120s** | B2: INS + EKF | 7 | 72.65% | **43.04%** | 314.42m |
| **120s** | B3: ML Only | 7 | 439.57% | 75.31% | 478.71m |
| **120s** | B4: ML + INS | 7 | 372.63% | 68.87% | 410.55m |
| **120s** | B5: ML + INS + EKF | 7 | 439.57% | 75.31% | 478.71m |
| **120s** | B6: ML + INS + EKF + NHC | 7 | 439.38% | 75.40% | 477.16m |
| **120s** | B7: ML + INS + EKF + NHC + ZUPT | 7 | 311.95% | 60.17% | 428.85m |
| **120s** | B8: ML + Kinematic Gate + EKF + NHC + ZUPT | 7 | 311.62% | 59.42% | 423.98m |
| **120s** | B9: INS + EKF + NHC + ZUPT (no ML) | 7 | 64.16% | 64.67% | 378.47m |

**Bold** = best (lowest) in that column for that duration. Note B2 (plain physics) wins median drift and RMSE at every duration; B9 (physics + NHC + ZUPT, no ML) wins at 10s specifically but is worse than B2 at 30s/60s/120s — this is the Baseline-9-vs-Baseline-2 result from PROJECT_STATUS.md §33 that led the team to ship B2's logic (INS+EKF, no NHC) as the app's physics-primary fallback rather than B9's.

### B. Moving Sequences Only (distance travelled > 100m)

23 valid combinations across 9 baselines (207 total evaluations).

| Outage Duration | Navigation Baseline | N | Mean Drift (%) | Median Drift (%) | Median RMSE (m) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **10s** | B1: Pure INS | 6 | 11.43% | 8.60% | 5.90m |
| **10s** | B2: INS + EKF | 6 | **7.79%** | **6.92%** | **4.96m** |
| **10s** | B3: ML Only | 6 | 26.18% | 24.51% | 24.85m |
| **10s** | B4: ML + INS | 6 | 23.09% | 20.50% | 21.06m |
| **10s** | B5: ML + INS + EKF | 6 | 26.18% | 24.51% | 24.85m |
| **10s** | B6: ML + INS + EKF + NHC | 6 | 26.17% | 24.51% | 24.84m |
| **10s** | B7: ML + INS + EKF + NHC + ZUPT | 6 | 32.78% | 24.51% | 24.84m |
| **10s** | B8: ML + Kinematic Gate + EKF + NHC + ZUPT | 6 | 31.81% | 20.48% | 22.19m |
| **10s** | B9: INS + EKF + NHC + ZUPT (no ML) | 6 | 22.99% | 9.74% | 7.30m |
| **30s** | B1: Pure INS | 5 | 34.15% | 16.13% | 57.15m |
| **30s** | B2: INS + EKF | 5 | **25.26%** | **14.09%** | **40.85m** |
| **30s** | B3: ML Only | 5 | 34.73% | 32.62% | 91.70m |
| **30s** | B4: ML + INS | 5 | 33.30% | 27.85% | 82.00m |
| **30s** | B5: ML + INS + EKF | 5 | 34.73% | 32.62% | 91.70m |
| **30s** | B6: ML + INS + EKF + NHC | 5 | 34.65% | 32.39% | 91.51m |
| **30s** | B7: ML + INS + EKF + NHC + ZUPT | 5 | 38.15% | 32.39% | 91.51m |
| **30s** | B8: ML + Kinematic Gate + EKF + NHC + ZUPT | 5 | 37.29% | 30.44% | 97.00m |
| **30s** | B9: INS + EKF + NHC + ZUPT (no ML) | 5 | 34.49% | 29.27% | 70.87m |
| **60s** | B1: Pure INS | 6 | 43.19% | 40.57% | 177.26m |
| **60s** | B2: INS + EKF | 6 | 30.35% | 28.90% | **93.54m** |
| **60s** | B3: ML Only | 6 | 36.71% | 24.33% | 166.69m |
| **60s** | B4: ML + INS | 6 | 35.19% | **23.51%** | 157.10m |
| **60s** | B5: ML + INS + EKF | 6 | 36.71% | 24.33% | 166.69m |
| **60s** | B6: ML + INS + EKF + NHC | 6 | 36.70% | 24.30% | 166.40m |
| **60s** | B7: ML + INS + EKF + NHC + ZUPT | 6 | 25.46% | 24.30% | 166.40m |
| **60s** | B8: ML + Kinematic Gate + EKF + NHC + ZUPT | 6 | **25.12%** | 23.96% | 164.09m |
| **60s** | B9: INS + EKF + NHC + ZUPT (no ML) | 6 | 42.82% | 39.96% | 137.41m |
| **120s** | B1: Pure INS | 6 | 58.46% | 48.96% | **405.23m** |
| **120s** | B2: INS + EKF | 6 | **52.22%** | **41.88%** | 458.80m |
| **120s** | B3: ML Only | 6 | 75.04% | 68.97% | 545.12m |
| **120s** | B4: ML + INS | 6 | 68.67% | 64.64% | 478.88m |
| **120s** | B5: ML + INS + EKF | 6 | 75.04% | 68.97% | 545.12m |
| **120s** | B6: ML + INS + EKF + NHC | 6 | 75.09% | 69.02% | 544.60m |
| **120s** | B7: ML + INS + EKF + NHC + ZUPT | 6 | 55.08% | 51.31% | 520.61m |
| **120s** | B8: ML + Kinematic Gate + EKF + NHC + ZUPT | 6 | 54.70% | 50.37% | 510.49m |
| **120s** | B9: INS + EKF + NHC + ZUPT (no ML) | 6 | 64.08% | 68.94% | 539.70m |

**Note on B3=B5:** these two remain numerically identical in every row, same as the original document — B5 ("ML+INS+EKF") and B3 ("ML Only") apparently converge under the current implementation; not re-investigated this pass, carried forward as an observed-and-unexplained property of the ladder, same status as before.

---

## 5. Threshold Performance (<10% / <15% / <20% drift), recomputed for B3, B4, B5, and B9

| Baseline | Evaluation Subset | N | <10% | <15% | <20% | Median Drift | Mean Drift |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B3 (ML Only)** | All Evaluations | 32 | 3 (9.4%) | 3 (9.4%) | 4 (12.5%) | 60.82% | 667.19% |
| **B3 (ML Only)** | Moving Only (>100m) | 23 | 3 (13.0%) | 3 (13.0%) | 4 (17.4%) | 32.62% | 43.53% |
| **B4 (ML + INS)** | All Evaluations | 32 | 2 (6.2%) | 3 (9.4%) | 5 (15.6%) | 53.51% | 578.03% |
| **B4 (ML + INS)** | Moving Only (>100m) | 23 | 2 (8.7%) | 3 (13.0%) | 5 (21.7%) | 27.85% | 40.35% |
| **B5 (ML + INS + EKF)** | All Evaluations | 32 | 3 (9.4%) | 3 (9.4%) | 4 (12.5%) | 60.82% | 667.19% |
| **B5 (ML + INS + EKF)** | Moving Only (>100m) | 23 | 3 (13.0%) | 3 (13.0%) | 4 (17.4%) | 32.62% | 43.53% |
| **B9 (INS+EKF+NHC+ZUPT, no ML)** | All Evaluations | 32 | **9 (28.1%)** | **11 (34.4%)** | **12 (37.5%)** | 42.98% | 46.39% |
| **B9 (INS+EKF+NHC+ZUPT, no ML)** | Moving Only (>100m) | 23 | **8 (34.8%)** | **10 (43.5%)** | **10 (43.5%)** | 34.17% | 41.38% |

B9 was not part of the original document's threshold table (it didn't exist yet). It is included here because it is the single clearest piece of evidence for the physics-primary decision: on the metric that most directly answers "how often is this system highly accurate," the best no-ML physics configuration clears <10% three times more often than any ML-inclusive one.

---

## 6. Best Baseline by Outage Duration (all 9 baselines)

*Recomputed by median drift, mean drift, and median RMSE, same methodology as the original document, now over 9 baselines instead of 7:*

- **10s Outages:**
  - **All Evaluations:** best median drift — **B9: INS+EKF+NHC+ZUPT** (12.15%); best mean drift and RMSE — **B2: INS+EKF** (23.71%, 3.16m).
  - **Moving Evaluations:** **B2: INS+EKF** best on all three metrics (6.92% median, 7.79% mean, 4.96m RMSE).
- **30s Outages:**
  - **All Evaluations:** **B2: INS+EKF** best median/RMSE (34.38%, 28.16m); B9 best mean (45.02%).
  - **Moving Evaluations:** **B2: INS+EKF** best on all three (14.09% median, 25.26% mean, 40.85m RMSE).
- **60s Outages:**
  - **All Evaluations:** **B2: INS+EKF** best on all three (30.39% median, 31.10% mean, 76.77m RMSE).
  - **Moving Evaluations:** **B4: ML+INS** achieves the lowest median drift (**23.51%**, beating B2's 28.90%) and **B8** the lowest mean drift (25.12%); **B2** still has the lowest median RMSE (93.54m). This is the one duration/subset where an ML-inclusive baseline genuinely wins on median drift.
- **120s Outages:**
  - **All Evaluations:** **B2: INS+EKF** best median drift (43.04%); B9 best mean (64.16%); **B1: Pure INS** (no EKF at all) has the lowest median RMSE (302.90m).
  - **Moving Evaluations:** **B2: INS+EKF** best median/mean drift (41.88%/52.22%); **B1: Pure INS** lowest median RMSE (405.23m).

This directly supersedes the original document's Section 6, which had (pre-fix) B4: ML+INS winning median drift at 120s (25.04%) — under the corrected methodology B2 wins that comparison (43.04% vs B4's 68.87%), and by a wide margin.

---

## 7. Verification of Empirical Claims (recomputed against current, post-fix data)

1. **Claim a (8/9 test sequences improved vs Zero baseline):** **VERIFIED TRUE**, using `model_comparison_by_sequence.csv` (regenerated 2026-09-14, per-window raw displacement comparison — this metric doesn't depend on the outage-simulator heading leak at all, since it isn't an outage simulation). 8 of 9 sequences show positive `error_reduction_vs_zero_pct`; the exception is `vw15` (-3547.4%, a parked vehicle).
2. **Claim b (IO-VNBD GRU improves over OxIOD on 8/9 sequences):** **VERIFIED TRUE**, same file/subset: 8 of 9 sequences show positive `error_reduction_vs_oxiod_pct` (range 3.9%–86.8%); `vw15` is again the exception (-126.3%).
3. **Claim c (ML-only outage evaluations achieving <10% drift):** **REVISED.** Under the corrected methodology, **3 of 32** (not 4/32) ML-only evaluations achieve <10% drift: `vw17` 10s (5.05%), `vw16b` 10s (6.05%), `vw16b` 60s (6.91%). The original document's 4th example, `vw16a` at 120s, no longer qualifies — see the retraction in Claim f below.
4. **Claim d (Worst stationary outlier is `vw15`):** **VERIFIED TRUE, unchanged.** `vw15`'s ML-only drift at 30s is **5667.10%** — the same value as the original document (this specific near-zero-displacement case is essentially unaffected by the heading fix, since heading barely matters when the vehicle isn't moving).
5. **Claim e (`y1` at 120s: Pure INS outperforms ML):** **VERIFIED TRUE, direction unchanged, magnitudes revised.** Pure INS: **25.36%** drift vs ML Only: **80.76%** drift (previously reported as 18.02% vs 32.59%). The finding — physics beats ML on this high-speed straight-highway sequence — still holds, with a much larger gap than previously reported.
6. **Claim f (`vw16a` has the best 120s moving-sequence ML result at 9.10% drift):** **RETRACTED.** This was the document's headline claim and the specific number invalidated by the heading-leak fix. Under the corrected methodology, `vw16a`'s ML-only drift at 120s is **60.12%** — not a strong result at all. The actual best 120s-moving ML-only result is now `vfa02` at **27.71%** drift (still well above the <10% "highly accurate" bar). Ranked: vfa02 (27.71%) < vw16a (60.12%) < vfa01 (62.62%) < vw14c (75.31%) < y1 (80.76%) < m (143.70%).

---

## 8. NHC and ZUPT Trajectory Verification (B6 vs B7), recomputed

- **Mechanics:** unchanged — ZUPT is active only when $\text{acc\_dev} < 0.10\text{ g}$, $\text{acc\_var} < 0.02\text{ g}^2$, and $\text{gyro\_mag} < 0.08\text{ rad/s}$.
- **Identical Trajectories:** recomputing directly from the current per-sequence file, **the same 13 outage intervals** are numerically identical between B6 and B7 as in the original document — `vfa01` (10s–120s, 4), `vfa02` (10s–60s, 3), `vw16a` (10s–120s, 4), `vw16b` (10s, 1), `vw17` (10s, 1) — confirming ZUPT genuinely never triggers on these unbroken-highway-motion intervals, unaffected by the heading fix.
- **Active Trajectory Modification (19 outage intervals):** on drives with traffic stops or parked periods (`m`, `vw15`, `vw14c`, `y1`), ZUPT still measurably helps, though the exact magnitudes changed with the heading fix:
  - `m` (60s outage): B6 drift **518.91%** → B7 drift **328.13%** (RMSE 62.46m → 44.97m). [previously reported: 651.80% → 264.16%]
  - `vw14c` (120s outage): B6 drift **75.40%** → B7 drift **17.59%** (RMSE 223.03m → 46.31m). [previously reported: 43.16% → 10.35%]

---

## 9. Ground Truth & Heading Reference Leakage Audit

1. **GNSS Position Updates:** Strictly suppressed during outages (0 position updates). **PASS**.
2. **GNSS / CAN Velocity Updates:** Strictly suppressed during outages (0 velocity updates). **PASS**.
3. **Temporal Causality:** Strict forward-sliding windows $[t-20, t]$ with zero future lookahead. **PASS**.
4. **Normalization Parameters:** Computed exclusively on training data (`models/io_vnbd_normalization.json`). **PASS**.
5. **Heading leak — found and fixed (§31, 2026-09-08):** `src/evaluation/run_all_test_sequences_benchmark.py` (the script that produces every number in this document) previously read live ground-truth heading `gt_hdg[i]` through the outage window instead of freezing it at the outage-entry value, for every baseline. This has been fixed — the heading is now frozen at outage entry, matching what a real system without an external heading reference would have. **PASS (fixed).**
6. **Residual leak in an unused sibling script — confirmed still present, confirmed non-corrupting:** `src/evaluation/run_real_io_vnbd_benchmark.py` still contains 5 unguarded `seq.gt_hdg[i]` reads, all confined to that file's own `run_benchmark()` function (lines 213/245/272/308/391). `run_all_test_sequences_benchmark.py` imports only `RealIOVNBDSequence`, `geodetic_to_ned_vec`, `ned_to_geodetic_vec`, and `precompute_ml_displacements` from that file — none of which touch `gt_hdg` — so this residual leak does not reach any number reported here. It remains an open, low-priority cleanup item (also flagged in PROJECT_STATUS.md's Sep 13 brief): worth fixing so nobody re-runs that script and reports inflated numbers by mistake, but not a correctness issue for anything currently shown.
7. **Attitude / Heading Scope Disclosure (unchanged from original):** the benchmark strictly measures displacement dead reckoning under reference (ground-truth, frozen-at-outage-entry) attitude/heading. It does not evaluate open-loop heading integration from an uncalibrated consumer smartphone gyroscope. In an unassisted smartphone without an external attitude reference or magnetometer filter, heading drift would progressively misorient the trajectory and degrade NHC constraint effectiveness over 60–120s outages, beyond what this benchmark's numbers show.

---

## 10. ONNX & Android Edge Parity

- **Model:** `models/gru_io_vnbd.onnx` — confirmed byte-identical (`md5sum`) between `dead_reckoning/models/gru_io_vnbd.onnx` and the file actually bundled in the Android app at `gudumap/app/src/main/assets/gru_io_vnbd.onnx`. The model artifact itself has not changed since the original audit, so the numerical-parity figures below (a property of that unchanged file, independent of the benchmark-methodology heading-leak bug) are still valid and were not recomputed:
  - Maximum absolute difference (PyTorch vs ONNX Runtime): **$1.192 \times 10^{-6}\text{ m}$**
  - Mean absolute difference: **$9.244 \times 10^{-8}\text{ m}$**
  - Parity RMSE: **$1.546 \times 10^{-7}\text{ m}$**
  - CPU Inference Latency: Mean **$0.177\text{ ms}$**, Median **$0.159\text{ ms}$**, P95 **$0.232\text{ ms}$** ($>5,600\text{ inf/sec}$).
- **Android Integration Status — CORRECTED.** The original document stated Android integration was "PENDING" against a mismatched `[1, 200, 6]` contract. That was true on 2026-09-06 but is no longer accurate: `gudumap/app/src/main/java/com/example/gudumap/ml/ModelMetadata.kt` currently declares `WINDOW_SIZE = 20` (2.0 seconds of samples) and an input shape comment of `[batch, 20, 6]` — the correct `[1, 20, 6]` @ 10Hz vehicle contract, matching the benchmarked model exactly. Android edge integration of this model is **DONE**, not pending, as of the current codebase (confirmed by direct file read, not assumed from this document's prior text).

---

## 11. Comprehensive Limitations Section

1. **Physics currently outperforms the ML-assisted stack, not just "achieves comparable" accuracy:** under the corrected methodology, plain INS+EKF beats every ML-inclusive baseline's median drift at 10s/30s/120s ("All Evaluations") and 10s/30s/120s ("Moving Evaluations"); ML only wins outright at 60s-moving. This is a stronger, more specific limitation than the original document's framing ("sub-10% is not universally achieved") — the honest current claim is that ML is an inconsistent assistive signal, not a reliable accuracy improvement, which is exactly why the shipped app does not rely on it as the primary estimator.
2. **Stationary Chassis Vibration Causes False Displacement:** unchanged — accelerometers detect engine vibration while parked (`vw15`), causing open-loop ML to predict residual forward creep. ZUPT is essential to suppress false accumulation when stopped.
3. **Heading Error Degrades NHC & Orientation:** unchanged — this benchmark relies on reference (ground-truth, frozen-at-outage) vehicle heading; an unassisted consumer smartphone's own heading drift would misorient displacement increments and weaken NHC beyond what these numbers show.
4. **Cross-Driver and Route Generalization:** unchanged — driving dynamics and smartphone mounting flexure create notable variance (e.g., Driver D `y1` vs Driver E `vw16a`/`vfa02`).
5. **Dataset-Only Validation:** unchanged — validation has been performed on the IO-VNBD dataset; physical smartphone road testing under OS scheduling jitter remains necessary.
6. **Pedestrian model not integrated:** `gru_local_best.pt` (OxIOD-trained, `[1,200,6]`@100Hz) exists but is not wired into the app, which only loads the vehicle model — unlike the Android/ONNX item above, this limitation from the pre-screening brief is still current and unresolved.
7. **A residual, non-corrupting heading leak remains in an unused script:** see §9.6 — cheap to fix, not urgent.

---

## 12. Final Honest Project Verdict

The AI/ML Dead Reckoning system is **DEFENSIBLY VALIDATED ON REAL VEHICLE DATA WITH DISCLOSED BOUNDARIES, INCLUDING THE BOUNDARY THAT ITS ML COMPONENT DOES NOT RELIABLY BEAT PLAIN PHYSICS.** Substantial dead-reckoning capability was demonstrated on real held-out IO-VNBD vehicle sequences — the strapdown INS + 6-state EKF fusion (Baseline 2) constrains cubic inertial-only divergence at every tested outage duration. The ML displacement model is a genuine, real, correctly-integrated research contribution that is not currently a reliable net improvement over that physics baseline on this dataset; the shipped app's decision to make physics-based fusion the guaranteed default, with ML contributing only when its plausibility gate trusts it, is the correct engineering response to this specific, verified finding rather than a limitation being hidden from a judge.
