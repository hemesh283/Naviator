# SIH26168 — Pre-Screening-2 Status Brief
**Prepared:** Sep 13, 2026 · **Screening deadline:** ~2 days out
**Scope:** honest current-state assessment against your 4 asks, before any implementation starts.

---

## Bottom line

Two of your four asks (map, and half of "other fixes") are in better shape than you'd expect. One ask (UI) is a real gap but a fast one to close. And one ask (ML accuracy) has a finding that changes the whole conversation: **right now, turning the ML model off and using plain physics (INS+EKF) is more accurate than your full ML pipeline, at every outage duration tested.** That's not a tuning nit — it's the single most important thing to decide on before the screening, because it determines what you actually demo.

Here is the evidence for each ask, then a recommended 2-day plan for you to greenlight or redirect.

---

## Ask 1 — "Highly accurate ML, including pedestrian data"

### The model in the app is real and wired up correctly
Good news first: the Android app (`gudumap`) bundles `assets/gru_io_vnbd.onnx` and runs it through ONNX Runtime with the correct `[1, 20, 6]` @ 10 Hz contract, matching `ModelMetadata.kt` exactly. This is the *same* model your 224-sequence IO-VNBD benchmark evaluated — the "two disconnected models, Android integration pending" problem flagged in your own `FINAL_AUDIT_SUMMARY.md` (Section 10–11, dated Sep 6) **has already been resolved** since that doc was written. Someone wired the benchmarked model into the app. That's one less fire to fight.

### The finding that matters: the ML model is currently making things worse
I re-pulled `real_benchmark_aggregate.csv` (regenerated Sep 8, after the heading-leak fix I shipped for your last presentation) and laid out median drift-% by outage duration across all 7 baselines:

| Duration | Pure INS | **INS+EKF (no ML)** | ML Only | ML+INS | ML+INS+EKF | +NHC | +NHC+ZUPT |
|---|---|---|---|---|---|---|---|
| 10s  | 19.0% | **13.0%** | 35.4% | 33.5% | 35.4% | 35.4% | 35.4% |
| 30s  | 62.6% | **34.4%** | 50.8% | 47.7% | 50.8% | 50.8% | 46.4% |
| 60s  | 51.5% | **30.4%** | 31.8% | 32.4% | 31.8% | 31.7% | 31.7% |
| 120s | 55.0% | **43.0%** | 75.3% | 68.9% | 75.3% | 75.4% | 60.2% |

At **every single duration**, plain physics (Baseline 2: strapdown INS + 6-state EKF, zero ML) beats every baseline that includes the ML displacement model. At 120s outages — your headline scenario — the full ML stack drifts 75% vs. 43% for physics alone. That's not noise (n=7–9 real IO-VNBD sequences per row, not synthetic).

Why: the deployed model (`gru_io_vnbd.onnx`) was trained purely on vehicle data (IO-VNBD, CAN-bus ground truth) and is a small GRU regressing raw displacement, not a residual correction — so on unseen vehicle sequences it's extrapolating hard rather than correcting the physics estimate. This isn't the heading-leak bug from last time; that's already fixed and reflected in this table. This is a genuine model-quality/generalization gap.

**This is precisely what "highly accurate ML" needs to fix — and it's a 2-day-realistic problem if you scope it right, not a full-retrain problem:**
- Fastest, lowest-risk option: gate the ML contribution by confidence/duration (e.g., trust ML less on longer horizons, or drop back to Baseline 7's NHC+ZUPT-only physics stack when ML disagrees strongly with the EKF prediction). This alone would likely put you near the 30–43% band across the board instead of 60–75% at 120s, using code you already have (all 7 baselines already exist in `baseline_ladder.py`).
- If there's time after that: revisit training (more regularization, residual-target reformulation instead of raw displacement, or a stronger held-out validation split) — but that's a stretch goal for 2 days, not the primary fix.

### Pedestrian data — real gap, not yet in the product
`gru_local_best.pt` (OxIOD-trained, pedestrian/handheld, `[1,200,6]` @ 100Hz) exists in `models/` but is **not used anywhere in the Android app** — the app only loads the vehicle model. So "including pedestrian data for better representation" is currently zero-coverage in the actual product: there's a pedestrian-trained model sitting unused, with an incompatible input contract (200 samples/100Hz vs. the app's 20 samples/10Hz pipeline) and it was never part of your rigorous benchmark either — it has no ATE/RTE/drift numbers you could show a judge.

Realistic in 2 days: this is not "unify two models into one" — that's a retrain. What's achievable is either (a) demo the pedestrian model as a clearly-labeled second, separate mode (different sensor sampling rate, different screen/toggle) with whatever eval numbers you can generate against OxIOD test sequences in the time you have, or (b) be upfront that pedestrian is roadmap, not shipped, and focus the 2 days on making the vehicle path actually accurate (see above) — which is the stronger demo either way.

---

## Ask 2 — "Better and futuristic UI"

Current state, read directly from the code: it's a **single screen** (`NavigationScreen.kt`, 844 lines), Jetpack Compose + Material3, structured as a stack of diagnostic cards — a confidence card, a GNSS/EKF/ML/map status card, sensor-active indicators, metric cards. It's functional and clearly built for engineering visibility (which is genuinely useful for a judge who wants to see internals), but it reads as a debug dashboard, not a polished consumer nav app. No animation, no dark/premium theming beyond default Material3, one screen total (no onboarding, no trip summary, no settings).

This is honestly the most tractable of your four asks in the time available — it's UI polish on top of a working data layer, not a correctness problem. Realistic 2-day scope: keep the single-screen structure (don't risk breaking the working ViewModel/state wiring) but give it a real design pass — a proper full-bleed map as the primary surface with the status cards as a collapsible overlay/bottom sheet instead of a scroll stack, a custom color theme, motion on state transitions (GNSS lost/recovered, mode switches), and a cleaner typographic hierarchy. That's achievable without touching navigation/ML logic.

---

## Ask 3 — "Highly detailed and accurate map with accurate live movement during GPS blackout"

This is in better shape than the other asks. Confirmed in code:
- Real offline map: `osmdroid` + a bundled `coimbatore.mbtiles` (14MB, real tiles, zoom 11–16, verified against the file's own metadata table) plus a vector road overlay (`coimbatore_roads.json`, 896KB) — not a placeholder.
- Live vehicle marker with heading-based rotation, updated from `NavigationViewModel` state.
- **During blackout it renders two separate trails** — a "naive" trail (raw INS-only) and a "corrected" trail (your fused EKF/ML estimate) — so a judge can visually see dead reckoning correcting itself in real time against the raw drift. That's a strong demo feature already built, not something you need to add.

What's actually missing for "highly detailed and accurate": the accuracy of the corrected trail is exactly the Ask 1 problem above — the map will faithfully render whatever the fusion pipeline outputs, and right now that output is worse than plain-physics at long outages. Fix the fusion (Ask 1), and this map gets more accurate for free. Coverage is also limited to Coimbatore only (single city's tiles bundled) — fine for a scripted demo route, but worth confirming your demo path actually falls inside that 14MB tile set before the screening.

---

## Ask 4 — Other fixes (found during this review, not asked for but load-bearing)

1. **`FINAL_AUDIT_SUMMARY.md` is now stale and would embarrass you if shown as-is.** It's dated Sep 6, written *before* my Sep 8 heading-leak fix, and its headline claim — "9.10% drift on 120s outages" — no longer matches reality. Current 120s numbers (table above) run 43–75% depending on baseline. If a judge who saw your last round's numbers cross-checks this document against a live demo, the mismatch is the kind of thing that costs credibility fast. This needs to be regenerated from the current CSVs (`real_benchmark_aggregate.csv`, mtime Sep 8) before the screening — I did not touch it yet, flagging it for your call on priority.
2. `run_real_io_vnbd_benchmark.py` still has the same ground-truth heading leak I fixed in `run_all_test_sequences_benchmark.py` (confirmed via grep — 5 unguarded `seq.gt_hdg[i]` reads). It doesn't feed your current results file, so it's not corrupting anything you're showing — but if anyone runs it and reports those numbers, they'll be the old inflated kind. Low priority, but cheap to fix.
3. `PROJECT_ROOT = Path("D:/dead_reckoning")` is hardcoded in two evaluation scripts and doesn't match your actual path (`D:\Projects\SIH_2026\dead_reckoning`). Hasn't bitten you yet apparently, but worth a one-line fix so a teammate re-running the benchmark doesn't silently fail or write to the wrong place.

---

## Recommended priority for the next 2 days

Given the timeline, in order of what actually moves the needle for a technically literate judge:

1. **Fix the ML-vs-physics regression (Ask 1's real finding).** This is the one thing that, if a judge asks "does your ML actually help," you currently cannot answer "yes" to with your own data. The confidence-gating approach is scoped to be fast.
2. **Regenerate `FINAL_AUDIT_SUMMARY.md`** (or clearly mark it superseded) so your own documentation doesn't contradict your live demo.
3. **UI pass** — full-bleed map + overlay card redesign, motion on state changes. High visual payoff, low technical risk.
4. **Pedestrian mode** — only if 1–3 leave slack; scope it as a clearly-labeled secondary demo path, not a merged model.
5. Cheap fixes (leak in the unused script, hardcoded path) — do these opportunistically, not on the critical path.

I haven't started implementation on any of this — flagging it all for your call on sequencing before I touch code, per your ask. Let me know which of these you want me to start on first.
