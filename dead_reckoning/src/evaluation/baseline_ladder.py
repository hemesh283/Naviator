"""Scientific Baseline Ladder for Vehicle Dead Reckoning Comparison.

Implements all 8 evaluation baselines:
  BASELINE 1: Pure inertial / dead reckoning (double integration of specific force)
  BASELINE 2: INS + EKF (kinematic INS propagation + EKF)
  BASELINE 3: ML displacement only (pure GRU inference displacement accumulation)
  BASELINE 4: ML + INS (coupled motion estimation)
  BASELINE 5: ML + INS + EKF (fused with GNSS outside outages)
  BASELINE 6: ML + INS + EKF + NHC (adding non-holonomic velocity constraints)
  BASELINE 7: ML + INS + EKF + NHC + ZUPT (adding multi-signal stationary pseudo-measurements)
  BASELINE 8: ML + Kinematic Gate + EKF + NHC + ZUPT (Baseline 7, but the raw ML
    displacement is first passed through a faithful port of
    DeadReckoningEngine.kt's ML kinematic plausibility gate -- see
    _run_baseline_8_gated / GATE_* constants below. Kept consistent with the
    same Baseline 8 added to run_all_test_sequences_benchmark.py, which is the
    script that actually produces this project's published benchmark CSVs;
    this module is used by the separate run_io_vnbd_benchmark.py CLI.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.inference.gru_inference import DeadReckoningInference
from src.navigation.coordinate_frames import CoordinateTransformer, MEAN_EARTH_RADIUS
from src.navigation.ekf import NavigationEKF
from src.navigation.nhc import NhcConstraint
from src.navigation.zupt import ZuptDetector
from src.preprocessing.resampler import ResampledVehicleStream

# ML Kinematic Plausibility Gate constants -- faithful port of
# DeadReckoningEngine.kt (gudumap Android app), companion object + the
# vehicle-mode branch of processWindowInference(). Same names, same values,
# kept consistent with run_all_test_sequences_benchmark.py's copy of these
# same constants. The pedestrian-safe fallback (MotionMode.CONSERVATIVE_MODE)
# in that file is NOT ported here -- this benchmark is vehicle data only.
GATE_MAX_SPEED_CHANGE_MPS2 = 4.0
GATE_MAX_PLAUSIBLE_SPEED_MPS = 50.0
GATE_STRIDE_DURATION_SEC = 1.0  # the ML window/stride duration used in the
                                # gate's own kinematic-distance formula --
                                # NOT this module's 100 Hz sample dt (0.01s).
GATE_TOLERANCE_M = 1.2
GATE_CLAMP_TOLERANCE_M = 0.15
GATE_MIN_EFFECTIVE_ACC_MPS2 = 0.20
GATE_LOW_ACCEL_THRESHOLD_MPS2 = 0.35
GATE_LOW_SPEED_THRESHOLD_MPS = 0.30


@dataclass
class TrajectorySolution:
    """Estimated trajectory solution from a baseline."""

    name: str
    lat: np.ndarray       # Shape: (M,)
    lon: np.ndarray       # Shape: (M,)
    speed_mps: np.ndarray # Shape: (M,)
    heading_deg: np.ndarray # Shape: (M,)


def _compute_effective_heading(gnss_hdg_100hz: np.ndarray, gnss_mask: np.ndarray) -> np.ndarray:
    """Heading stream with the GNSS-outage leak removed.

    BUG THIS FIXES: every baseline below used to read ``stream.gnss_hdg_100hz``
    unconditionally, every sample, to rotate the INS/ML local-frame motion
    increment into NED -- including deep inside a simulated GNSS blackout.
    During a real blackout there is no live GNSS course to read, so this was
    silently leaking the withheld ground-truth heading into every "GNSS-denied"
    evaluation (all 7 baselines, since even Baseline 1 reads it unconditionally).

    That leak is also the most likely cause of the "trajectory jumps on a
    slight altitude change" symptom: whenever the live/real heading value is
    noisy or stale right at a motion event (a grade, a bump -- exactly where
    altitude changes and GPS course-over-ground gets noisiest), the ML
    displacement gets rotated by a jittery heading and the corrected path
    visibly jukes sideways even though the underlying displacement estimate
    didn't change much.

    Fix: freeze heading at its last known-good (pre-outage) GNSS value for
    the duration of the blackout, instead of continuing to read GNSS course.
    This removes the leak and, just as importantly, removes the per-sample
    heading jitter that was causing the spurious sideways motion -- a frozen
    heading means the rotation direction stays stable and coherent across an
    entire outage instead of chasing a signal that shouldn't be visible.
    """
    eff = gnss_hdg_100hz.astype(np.float64).copy()
    if len(eff) == 0:
        return eff
    last_good = eff[0]
    for i in range(len(eff)):
        if gnss_mask[i]:
            last_good = eff[i]
        else:
            eff[i] = last_good
    return eff


def _compute_blackout_entry_speed_and_elapsed(
    gnss_spd_100hz: np.ndarray, gnss_mask: np.ndarray, dt_sample: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample (blackoutEntrySpeedMps, elapsedBlackoutSec) for the ML Kinematic
    Plausibility Gate's baseSpeed envelope -- mirrors DeadReckoningEngine.kt's
    setBlackoutMode(true) capturing blackoutEntrySpeedMps/blackoutEntryTimestampNs
    ONCE at the moment GNSS is lost, then never touching them again until GNSS
    returns. Unlike run_all_test_sequences_benchmark.py (which evaluates one
    discrete, pre-planned outage interval at a time), this module's gnss_mask can
    toggle multiple times across a single continuous stream, so this is computed
    as a single forward pass (same style as _compute_effective_heading above):
    entry speed/elapsed reset every time the mask transitions available -> lost,
    and elapsed keeps growing for as long as the mask stays continuously lost.
    """
    n = len(gnss_mask)
    entry_speed = np.zeros(n, dtype=np.float64)
    elapsed = np.zeros(n, dtype=np.float64)
    if n == 0:
        return entry_speed, elapsed

    last_good_speed = float(gnss_spd_100hz[0])
    in_outage = False
    time_since_entry = 0.0
    for i in range(n):
        if gnss_mask[i]:
            last_good_speed = float(gnss_spd_100hz[i])
            in_outage = False
            time_since_entry = 0.0
        else:
            if not in_outage:
                in_outage = True
                time_since_entry = 0.0
            else:
                time_since_entry += dt_sample
        entry_speed[i] = last_good_speed
        elapsed[i] = time_since_entry
    return entry_speed, elapsed


class BaselineLadderEvaluator:
    """Evaluates the ladder of navigation algorithms under identical sensor streams."""

    def __init__(
        self,
        inference_engine: DeadReckoningInference,
        zupt_detector: Optional[ZuptDetector] = None,
        nhc_constraint: Optional[NhcConstraint] = None,
    ) -> None:
        self.infer = inference_engine
        self.zupt = zupt_detector or ZuptDetector()
        self.nhc = nhc_constraint or NhcConstraint()
        self.transformer = CoordinateTransformer()

    def run_all_baselines(
        self,
        stream: ResampledVehicleStream,
        gnss_mask: np.ndarray,
    ) -> Dict[str, TrajectorySolution]:
        """Execute all 7 baselines across the resampled vehicle stream.

        Args:
            stream: 100 Hz resampled vehicle stream.
            gnss_mask: Boolean array where True = GNSS available, False = Outage.

        Returns:
            Dictionary of baseline name -> TrajectorySolution.
        """
        # Precompute ML window displacements for efficiency
        ml_displacements = self._precompute_ml_displacements(stream.windows)

        # Heading with the outage leak removed -- see _compute_effective_heading.
        # Passed into every baseline below instead of stream.gnss_hdg_100hz directly.
        eff_hdg = _compute_effective_heading(stream.gnss_hdg_100hz, gnss_mask)

        solutions = {}
        solutions["BASELINE 1: Pure INS"] = self._run_baseline_1_pure_ins(stream, gnss_mask, eff_hdg)
        solutions["BASELINE 2: INS + EKF"] = self._run_baseline_2_ins_ekf(stream, gnss_mask, eff_hdg)
        solutions["BASELINE 3: ML Only"] = self._run_baseline_3_ml_only(stream, gnss_mask, ml_displacements, eff_hdg)
        solutions["BASELINE 4: ML + INS"] = self._run_baseline_4_ml_ins(stream, gnss_mask, ml_displacements, eff_hdg)
        solutions["BASELINE 5: ML + INS + EKF"] = self._run_baseline_5_ekf_fusion(
            stream, gnss_mask, ml_displacements, eff_hdg, use_nhc=False, use_zupt=False
        )
        solutions["BASELINE 6: ML + INS + EKF + NHC"] = self._run_baseline_5_ekf_fusion(
            stream, gnss_mask, ml_displacements, eff_hdg, use_nhc=True, use_zupt=False
        )
        solutions["BASELINE 7: ML + INS + EKF + NHC + ZUPT"] = self._run_baseline_5_ekf_fusion(
            stream, gnss_mask, ml_displacements, eff_hdg, use_nhc=True, use_zupt=True
        )
        solutions["BASELINE 8: ML + Kinematic Gate + EKF + NHC + ZUPT"] = self._run_baseline_8_gated(
            stream, gnss_mask, ml_displacements, eff_hdg
        )

        return solutions

    def _precompute_ml_displacements(self, windows: np.ndarray) -> np.ndarray:
        """Run batch inference over all 200-sample windows."""
        if len(windows) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return self.infer.predict_batch(windows) # Shape: (num_windows, 3)

    def _run_baseline_1_pure_ins(self, stream: ResampledVehicleStream, gnss_mask: np.ndarray, eff_hdg: np.ndarray) -> TrajectorySolution:
        """Baseline 1: Pure inertial double integration in navigation frame."""
        n = len(stream.t_100hz)
        dt = 0.01
        p_n = np.zeros(n)
        p_e = np.zeros(n)
        v_n = np.zeros(n)
        v_e = np.zeros(n)

        # Initialize velocity from initial GNSS speed & heading
        init_spd = float(stream.gnss_spd_100hz[0])
        init_hdg = float(stream.gnss_hdg_100hz[0])
        v_n[0] = init_spd * np.cos(np.deg2rad(init_hdg))
        v_e[0] = init_spd * np.sin(np.deg2rad(init_hdg))

        origin_lat = stream.gnss_lat_100hz[0]
        origin_lon = stream.gnss_lon_100hz[0]

        for i in range(1, n):
            # was: stream.gnss_hdg_100hz[i] -- leaked real GNSS heading during outages
            hdg = eff_hdg[i]
            R_bn = CoordinateTransformer.heading_to_dcm(hdg)
            # Accelerations in m/s^2
            a_body = stream.features_100hz[i, 0:3] * 9.80665
            a_ned = R_bn @ a_body

            if gnss_mask[i]:
                # Outside outage, track GNSS
                pn_g, pe_g = CoordinateTransformer.geodetic_to_ned(
                    stream.gnss_lat_100hz[i], stream.gnss_lon_100hz[i], origin_lat, origin_lon
                )
                p_n[i] = pn_g
                p_e[i] = pe_g
                spd = float(stream.gnss_spd_100hz[i])
                v_n[i] = spd * np.cos(np.deg2rad(hdg))
                v_e[i] = spd * np.sin(np.deg2rad(hdg))
            else:
                # During outage, double integration
                v_n[i] = v_n[i - 1] + a_ned[0] * dt
                v_e[i] = v_e[i - 1] + a_ned[1] * dt
                p_n[i] = p_n[i - 1] + v_n[i] * dt
                p_e[i] = p_e[i - 1] + v_e[i] * dt

        lat = np.zeros(n)
        lon = np.zeros(n)
        for i in range(n):
            lat[i], lon[i] = CoordinateTransformer.ned_to_geodetic(p_n[i], p_e[i], origin_lat, origin_lon)

        spd = np.sqrt(v_n ** 2 + v_e ** 2)
        return TrajectorySolution("BASELINE 1: Pure INS", lat, lon, spd, stream.gnss_hdg_100hz)

    def _run_baseline_2_ins_ekf(self, stream: ResampledVehicleStream, gnss_mask: np.ndarray, eff_hdg: np.ndarray) -> TrajectorySolution:
        """Baseline 2: INS Kinematic propagation with 6-state EKF."""
        n = len(stream.t_100hz)
        dt = 0.01
        origin_lat = stream.gnss_lat_100hz[0]
        origin_lon = stream.gnss_lon_100hz[0]

        ekf = NavigationEKF()
        init_spd = float(stream.gnss_spd_100hz[0])
        init_hdg = float(stream.gnss_hdg_100hz[0])
        vn = init_spd * np.cos(np.deg2rad(init_hdg))
        ve = init_spd * np.sin(np.deg2rad(init_hdg))
        ekf.initialize(0.0, 0.0, 0.0, vn, ve, 0.0)

        lat = np.zeros(n)
        lon = np.zeros(n)
        spd = np.zeros(n)

        for i in range(n):
            # was: stream.gnss_hdg_100hz[i] -- leaked real GNSS heading during outages
            hdg = eff_hdg[i]
            R_bn = CoordinateTransformer.heading_to_dcm(hdg)
            a_body = stream.features_100hz[i, 0:3] * 9.80665
            a_ned = R_bn @ a_body

            # Predict using kinematic acceleration
            v_curr = ekf.velocity_ned
            dp = v_curr * dt + 0.5 * a_ned * (dt ** 2)
            ekf.predict(dp, dt)

            # GNSS update when available
            if gnss_mask[i]:
                pn, pe = CoordinateTransformer.geodetic_to_ned(
                    stream.gnss_lat_100hz[i], stream.gnss_lon_100hz[i], origin_lat, origin_lon
                )
                ekf.update_gnss_position(pn, pe, 0.0)
                g_spd = float(stream.gnss_spd_100hz[i])
                ekf.update_gnss_velocity(
                    g_spd * np.cos(np.deg2rad(hdg)),
                    g_spd * np.sin(np.deg2rad(hdg)),
                    0.0
                )

            pos = ekf.position_ned
            lat[i], lon[i] = CoordinateTransformer.ned_to_geodetic(pos[0], pos[1], origin_lat, origin_lon)
            v = ekf.velocity_ned
            spd[i] = np.sqrt(v[0] ** 2 + v[1] ** 2)

        return TrajectorySolution("BASELINE 2: INS + EKF", lat, lon, spd, stream.gnss_hdg_100hz)

    def _run_baseline_3_ml_only(
        self,
        stream: ResampledVehicleStream,
        gnss_mask: np.ndarray,
        ml_displacements: np.ndarray,
        eff_hdg: np.ndarray,
    ) -> TrajectorySolution:
        """Baseline 3: ML displacement accumulation along heading."""
        n = len(stream.t_100hz)
        origin_lat = stream.gnss_lat_100hz[0]
        origin_lon = stream.gnss_lon_100hz[0]

        p_n = np.zeros(n)
        p_e = np.zeros(n)
        spd = np.zeros(n)

        # Distribute 1s window ML displacements evenly across 100 Hz samples
        # Each window covers 100 samples stride (1.0 s)
        step_dp_ned = np.zeros((n, 3))
        for w_idx, s_idx in enumerate(stream.window_start_indices):
            e_idx = stream.window_end_indices[w_idx]
            stride_len = max(1, e_idx - s_idx + 1)
            # ML output is [dx, dy, dz] in local body frame
            dx, dy, dz = ml_displacements[w_idx]
            # was: stream.gnss_hdg_100hz[s_idx] -- leaked real GNSS heading during outages
            hdg = float(eff_hdg[s_idx])
            R_bn = CoordinateTransformer.heading_to_dcm(hdg)
            dp_ned = R_bn @ np.array([dx, dy, dz])
            # Per-sample displacement
            dp_per_sample = dp_ned / stride_len
            step_dp_ned[s_idx : e_idx + 1] = dp_per_sample

        for i in range(1, n):
            if gnss_mask[i]:
                pn, pe = CoordinateTransformer.geodetic_to_ned(
                    stream.gnss_lat_100hz[i], stream.gnss_lon_100hz[i], origin_lat, origin_lon
                )
                p_n[i] = pn
                p_e[i] = pe
                spd[i] = stream.gnss_spd_100hz[i]
            else:
                p_n[i] = p_n[i - 1] + step_dp_ned[i, 0]
                p_e[i] = p_e[i - 1] + step_dp_ned[i, 1]
                spd[i] = float(np.sqrt(step_dp_ned[i, 0] ** 2 + step_dp_ned[i, 1] ** 2) / 0.01)

        lat = np.zeros(n)
        lon = np.zeros(n)
        for i in range(n):
            lat[i], lon[i] = CoordinateTransformer.ned_to_geodetic(p_n[i], p_e[i], origin_lat, origin_lon)

        return TrajectorySolution("BASELINE 3: ML Only", lat, lon, spd, stream.gnss_hdg_100hz)

    def _run_baseline_4_ml_ins(
        self,
        stream: ResampledVehicleStream,
        gnss_mask: np.ndarray,
        ml_displacements: np.ndarray,
        eff_hdg: np.ndarray,
    ) -> TrajectorySolution:
        """Baseline 4: Blended kinematic ML + INS motion."""
        sol_ins = self._run_baseline_1_pure_ins(stream, gnss_mask, eff_hdg)
        sol_ml = self._run_baseline_3_ml_only(stream, gnss_mask, ml_displacements, eff_hdg)

        # 80% ML displacement + 20% kinematic INS
        alpha = 0.85
        lat = alpha * sol_ml.lat + (1.0 - alpha) * sol_ins.lat
        lon = alpha * sol_ml.lon + (1.0 - alpha) * sol_ins.lon
        spd = alpha * sol_ml.speed_mps + (1.0 - alpha) * sol_ins.speed_mps

        return TrajectorySolution("BASELINE 4: ML + INS", lat, lon, spd, stream.gnss_hdg_100hz)

    def _run_baseline_5_ekf_fusion(
        self,
        stream: ResampledVehicleStream,
        gnss_mask: np.ndarray,
        ml_displacements: np.ndarray,
        eff_hdg: np.ndarray,
        use_nhc: bool = False,
        use_zupt: bool = False,
    ) -> TrajectorySolution:
        """Baselines 5, 6, 7: EKF fusion with ML motion updates, optional NHC & ZUPT."""
        n = len(stream.t_100hz)
        dt = 0.01
        origin_lat = stream.gnss_lat_100hz[0]
        origin_lon = stream.gnss_lon_100hz[0]

        ekf = NavigationEKF()
        init_spd = float(stream.gnss_spd_100hz[0])
        init_hdg = float(stream.gnss_hdg_100hz[0])
        vn = init_spd * np.cos(np.deg2rad(init_hdg))
        ve = init_spd * np.sin(np.deg2rad(init_hdg))
        ekf.initialize(0.0, 0.0, 0.0, vn, ve, 0.0)

        # Precompute per-sample ML displacement
        step_dp_ned = np.zeros((n, 3))
        for w_idx, s_idx in enumerate(stream.window_start_indices):
            e_idx = stream.window_end_indices[w_idx]
            stride_len = max(1, e_idx - s_idx + 1)
            dx, dy, dz = ml_displacements[w_idx]
            # was: stream.gnss_hdg_100hz[s_idx] -- leaked real GNSS heading during outages
            hdg = float(eff_hdg[s_idx])
            R_bn = CoordinateTransformer.heading_to_dcm(hdg)
            dp_ned = R_bn @ np.array([dx, dy, dz])
            step_dp_ned[s_idx : e_idx + 1] = dp_ned / stride_len

        lat = np.zeros(n)
        lon = np.zeros(n)
        spd = np.zeros(n)

        name = "BASELINE 5: ML + INS + EKF"
        if use_nhc and not use_zupt:
            name = "BASELINE 6: ML + INS + EKF + NHC"
        elif use_nhc and use_zupt:
            name = "BASELINE 7: ML + INS + EKF + NHC + ZUPT"

        for i in range(n):
            # was: stream.gnss_hdg_100hz[i] -- leaked real GNSS heading during outages
            # (this is the value fed to NHC's lateral-velocity constraint below)
            hdg = float(eff_hdg[i])
            dp = step_dp_ned[i]

            # EKF Prediction with ML displacement increment
            ekf.predict(dp, dt)

            # 1. ZUPT pseudo-measurement if enabled and stationary
            if use_zupt and i >= 15:
                recent_acc = stream.features_100hz[i - 15 : i, 0:3]
                recent_gyro = stream.features_100hz[i - 15 : i, 3:6]
                g_spd = float(stream.gnss_spd_100hz[i]) if gnss_mask[i] else None
                if self.zupt.is_stationary(recent_acc, recent_gyro, g_spd):
                    ekf.update_zupt(noise_mps=0.05)

            # 2. NHC pseudo-measurement if enabled
            if use_nhc:
                ekf.update_nhc(hdg, noise_lat_mps=0.15, noise_vert_mps=0.15)

            # 3. GNSS measurement update when available
            if gnss_mask[i]:
                pn, pe = CoordinateTransformer.geodetic_to_ned(
                    stream.gnss_lat_100hz[i], stream.gnss_lon_100hz[i], origin_lat, origin_lon
                )
                ekf.update_gnss_position(pn, pe, 0.0, noise_m=2.5)
                g_spd = float(stream.gnss_spd_100hz[i])
                ekf.update_gnss_velocity(
                    g_spd * np.cos(np.deg2rad(hdg)),
                    g_spd * np.sin(np.deg2rad(hdg)),
                    0.0,
                    noise_mps=0.2
                )

            pos = ekf.position_ned
            lat[i], lon[i] = CoordinateTransformer.ned_to_geodetic(pos[0], pos[1], origin_lat, origin_lon)
            v = ekf.velocity_ned
            spd[i] = np.sqrt(v[0] ** 2 + v[1] ** 2)

        return TrajectorySolution(name, lat, lon, spd, stream.gnss_hdg_100hz)

    def _run_baseline_8_gated(
        self,
        stream: ResampledVehicleStream,
        gnss_mask: np.ndarray,
        ml_displacements: np.ndarray,
        eff_hdg: np.ndarray,
    ) -> TrajectorySolution:
        """Baseline 8: ML + Kinematic Gate + EKF + NHC + ZUPT.

        Identical to Baseline 7 (_run_baseline_5_ekf_fusion with use_nhc=use_zupt=
        True) except the raw ML displacement is passed through the ML Kinematic
        Plausibility Gate -- a faithful port of DeadReckoningEngine.kt's
        processWindowInference() vehicle-mode gate (see GATE_* constants above) --
        before being rotated into NED and fused. Runs once per ML WINDOW (matching
        the real app's once-per-1.0s-stride inference cadence), not once per 100 Hz
        sample; the resulting (possibly clamped/rejected) window-level displacement
        is then spread evenly across that window's samples, same as every other
        baseline here.

        NOT ported: DeadReckoningEngine's separate `isNavStationary` pre-check (a
        multi-signal ZUPT classifier evaluated before the kinematic gate even
        runs) -- out of scope, same reasoning as
        run_all_test_sequences_benchmark.py's copy of this gate: not part of the
        task's own gate-formula summary, and substantially subsumed by this gate's
        own maxHorizAcc < 0.35 reject condition anyway.
        """
        n = len(stream.t_100hz)
        dt = 0.01
        origin_lat = stream.gnss_lat_100hz[0]
        origin_lon = stream.gnss_lon_100hz[0]

        ekf = NavigationEKF()
        init_spd = float(stream.gnss_spd_100hz[0])
        init_hdg = float(stream.gnss_hdg_100hz[0])
        vn = init_spd * np.cos(np.deg2rad(init_hdg))
        ve = init_spd * np.sin(np.deg2rad(init_hdg))
        ekf.initialize(0.0, 0.0, 0.0, vn, ve, 0.0)

        # Per-sample baseSpeed envelope inputs -- see
        # _compute_blackout_entry_speed_and_elapsed for why this must be a single
        # forward pass rather than reading the EKF's own live velocity (that would
        # reintroduce the exact self-referential feedback loop
        # DeadReckoningEngine.kt's Fix 1 exists to avoid).
        entry_speed, elapsed_blackout = _compute_blackout_entry_speed_and_elapsed(
            stream.gnss_spd_100hz, gnss_mask, dt_sample=dt
        )

        # Precompute per-sample GATED ML displacement (mirrors the ungated
        # precompute loop above, but runs every window's raw prediction through
        # the kinematic gate first).
        step_dp_ned = np.zeros((n, 3))
        for w_idx, s_idx in enumerate(stream.window_start_indices):
            e_idx = stream.window_end_indices[w_idx]
            stride_len = max(1, e_idx - s_idx + 1)
            dx, dy, dz = ml_displacements[w_idx]
            raw_pred = np.array([dx, dy, dz], dtype=np.float64)
            raw_mag = float(np.linalg.norm(raw_pred))

            a_body_window = stream.features_100hz[s_idx : e_idx + 1, 0:3] * 9.80665
            if len(a_body_window) > 0:
                a_h = np.sqrt(a_body_window[:, 0] ** 2 + a_body_window[:, 1] ** 2)
                max_horiz_acc = float(np.max(a_h))
            else:
                max_horiz_acc = 0.0

            base_speed_envelope = min(
                entry_speed[s_idx] + GATE_MAX_SPEED_CHANGE_MPS2 * elapsed_blackout[s_idx],
                GATE_MAX_PLAUSIBLE_SPEED_MPS,
            )
            effective_acc = max(max_horiz_acc, GATE_MIN_EFFECTIVE_ACC_MPS2)
            kinematic_dist = (
                base_speed_envelope * GATE_STRIDE_DURATION_SEC
                + 0.5 * effective_acc * GATE_STRIDE_DURATION_SEC ** 2
            )
            max_plausible_dist = kinematic_dist + GATE_TOLERANCE_M

            if (
                max_horiz_acc < GATE_LOW_ACCEL_THRESHOLD_MPS2
                and base_speed_envelope < GATE_LOW_SPEED_THRESHOLD_MPS
                and raw_mag > max_plausible_dist
            ):
                gated_local = np.zeros(3, dtype=np.float64)
            elif raw_mag > max_plausible_dist:
                max_clamped_dist = kinematic_dist + GATE_CLAMP_TOLERANCE_M
                scale = (max_clamped_dist / raw_mag) if raw_mag > 0.001 else 0.0
                gated_local = raw_pred * scale
            else:
                gated_local = raw_pred

            hdg = float(eff_hdg[s_idx])
            R_bn = CoordinateTransformer.heading_to_dcm(hdg)
            dp_ned = R_bn @ gated_local
            step_dp_ned[s_idx : e_idx + 1] = dp_ned / stride_len

        lat = np.zeros(n)
        lon = np.zeros(n)
        spd = np.zeros(n)

        for i in range(n):
            hdg = float(eff_hdg[i])
            dp = step_dp_ned[i]

            ekf.predict(dp, dt)

            if i >= 15:
                recent_acc = stream.features_100hz[i - 15 : i, 0:3]
                recent_gyro = stream.features_100hz[i - 15 : i, 3:6]
                g_spd = float(stream.gnss_spd_100hz[i]) if gnss_mask[i] else None
                if self.zupt.is_stationary(recent_acc, recent_gyro, g_spd):
                    ekf.update_zupt(noise_mps=0.05)

            ekf.update_nhc(hdg, noise_lat_mps=0.15, noise_vert_mps=0.15)

            if gnss_mask[i]:
                pn, pe = CoordinateTransformer.geodetic_to_ned(
                    stream.gnss_lat_100hz[i], stream.gnss_lon_100hz[i], origin_lat, origin_lon
                )
                ekf.update_gnss_position(pn, pe, 0.0, noise_m=2.5)
                g_spd = float(stream.gnss_spd_100hz[i])
                ekf.update_gnss_velocity(
                    g_spd * np.cos(np.deg2rad(hdg)),
                    g_spd * np.sin(np.deg2rad(hdg)),
                    0.0,
                    noise_mps=0.2
                )

            pos = ekf.position_ned
            lat[i], lon[i] = CoordinateTransformer.ned_to_geodetic(pos[0], pos[1], origin_lat, origin_lon)
            v = ekf.velocity_ned
            spd[i] = np.sqrt(v[0] ** 2 + v[1] ** 2)

        return TrajectorySolution("BASELINE 8: ML + Kinematic Gate + EKF + NHC + ZUPT", lat, lon, spd, stream.gnss_hdg_100hz)
