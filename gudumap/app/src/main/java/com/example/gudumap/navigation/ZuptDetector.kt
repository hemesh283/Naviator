package com.example.gudumap.navigation

import android.util.Log
import com.example.gudumap.sensor.ImuSample
import kotlin.math.sqrt

enum class NavMotionState {
    STATIONARY,
    ROTATING_IN_PLACE,
    MOVING
}

private const val TAG = "Gudumap:ZuptDetector"

/**
 * Multi-signal Stationary & In-Place Rotation Detector for Zero-Velocity Updates (ZUPT).
 *
 * Combines:
 * 1. Accelerometer 3D magnitude and variance statistics
 * 2. Horizontal translational acceleration magnitude and variance
 * 3. Gyroscope magnitude and variance statistics
 * 4. Optional GNSS speed
 *
 * Identifies:
 * - STATIONARY: Phone lying completely still on table or resting still in hand.
 * - ROTATING_IN_PLACE: Phone tilted, turned, or handled in place (gyro active, but no horizontal translation).
 * - MOVING: Genuine pedestrian walking or vehicular locomotion.
 *
 * Modular: Can be enabled/disabled dynamically.
 * Configurable: All detection thresholds can be tuned at runtime.
 */
class ZuptDetector(
    var isEnabled: Boolean = true,
    var accMagnitudeThreshold: Float = 0.25f,         // m/s^2 (linear acceleration)
    var accVarianceThreshold: Float = 0.04f,           // (m/s^2)^2
    var horizontalAccThreshold: Float = 0.35f,        // m/s^2 (horizontal translational acceleration)
    var horizontalAccVarianceThreshold: Float = 0.05f,// (m/s^2)^2
    var gyroMagnitudeThreshold: Float = 0.10f,         // rad/s
    var gyroVarianceThreshold: Float = 0.01f,          // (rad/s)^2
    var gnssSpeedThreshold: Float = 0.30f,             // m/s
    // §52: these used to be raw sample COUNTS (`minConsecutiveSamples = 4, "~400ms at 10Hz"`),
    // but `update()` is actually driven by `DeadReckoningEngine.addSensorSample()`, which itself
    // runs once per raw accelerometer OR gyroscope callback (`NavigationEngine.onSensorStep()`,
    // both registered at `SENSOR_DELAY_GAME` -- roughly 50Hz each, so combined this runs far
    // closer to ~100Hz than the assumed 10Hz). 4 samples was therefore actually confirming
    // stillness after roughly 40-80ms of low readings, not the intended 400ms. A single
    // pedestrian stride's brief low-acceleration moment (between a step's propulsion and
    // braking phases) comfortably fits inside 40-80ms -- which is exactly what let a phone
    // carried steadily while walking latch into STATIONARY and stay there (confirmed live: a
    // blackout-mode test session showing the app's own "Motion: STATIONARY" indicator and
    // "DR Distance: 0.0 m" the entire time the person was actually walking). Switched to real
    // elapsed time, measured from each sample's own timestamp, so the confirmation window is
    // whatever duration it says regardless of actual callback rate -- this restores the
    // originally documented/intended ~400ms debounce instead of guessing a new one, and applies
    // equally to vehicle mode (making its debounce more correct too, not just pedestrian mode).
    var minStationaryDurationMs: Float = 400f,
    var minRotatingDurationMs: Float = 200f,
    private val historyWindowSize: Int = 20
) {

    private val accMagHistory = FloatArray(historyWindowSize)
    private val accHorizHistory = FloatArray(historyWindowSize)
    private val gyroMagHistory = FloatArray(historyWindowSize)
    private var historyCount = 0
    private var historyIndex = 0

    // 0L = "not currently inside this condition" -- real sample timestamps (System.nanoTime()
    // via ImuSample.timestampNs) are never actually 0 in practice, so this sentinel is safe.
    private var stationaryConditionStartNs = 0L
    private var rotatingConditionStartNs = 0L
    private var isStationaryState = false

    var motionState: NavMotionState = NavMotionState.STATIONARY
        private set

    /**
     * Feed an incoming IMU sample and optional GNSS speed to update stationary state.
     *
     * @param sample Resampled or raw ImuSample
     * @param gnssSpeed Optional GNSS speed in m/s (null if not available)
     * @return true if stationary conditions are met and detector is enabled
     */
    fun update(sample: ImuSample, gnssSpeed: Float? = null): Boolean {
        if (!isEnabled) {
            isStationaryState = false
            motionState = NavMotionState.MOVING
            stationaryConditionStartNs = 0L
            rotatingConditionStartNs = 0L
            return false
        }

        val accMag = sqrt(sample.ax * sample.ax + sample.ay * sample.ay + sample.az * sample.az)
        val accHoriz = sqrt(sample.ax * sample.ax + sample.ay * sample.ay)
        val gyroMag = sqrt(sample.gx * sample.gx + sample.gy * sample.gy + sample.gz * sample.gz)

        // Store into histories
        accMagHistory[historyIndex] = accMag
        accHorizHistory[historyIndex] = accHoriz
        gyroMagHistory[historyIndex] = gyroMag
        historyIndex = (historyIndex + 1) % historyWindowSize
        if (historyCount < historyWindowSize) {
            historyCount++
        }

        // Calculate variances
        val accVar = calculateVariance(accMagHistory, historyCount)
        val accHorizVar = calculateVariance(accHorizHistory, historyCount)
        val gyroVar = calculateVariance(gyroMagHistory, historyCount)

        // 1. Evaluate full stationary conditions (table, or still in hand)
        val accCondition = (accMag < accMagnitudeThreshold) && (accVar < accVarianceThreshold)
        val gyroCondition = (gyroMag < gyroMagnitudeThreshold) && (gyroVar < gyroVarianceThreshold)
        val gnssCondition = gnssSpeed?.let { it < gnssSpeedThreshold } ?: true

        val instantStationary = accCondition && gyroCondition && gnssCondition

        // 2. Evaluate in-place rotation / handling without translational locomotion
        // Condition: gyro is active (turning, tilting), horizontal acceleration is low and steady, GNSS speed indicates no travel
        val gyroActive = (gyroMag >= gyroMagnitudeThreshold) || (gyroVar >= gyroVarianceThreshold)
        val horizAccLow = (accHoriz < horizontalAccThreshold) && (accHorizVar < horizontalAccVarianceThreshold)
        val instantRotatingInPlace = gyroActive && horizAccLow && gnssCondition

        val previousMotionState = motionState

        if (instantStationary) {
            if (stationaryConditionStartNs == 0L) stationaryConditionStartNs = sample.timestampNs
            rotatingConditionStartNs = 0L
            val elapsedMs = (sample.timestampNs - stationaryConditionStartNs) / 1_000_000f
            if (elapsedMs >= minStationaryDurationMs) {
                motionState = NavMotionState.STATIONARY
                isStationaryState = true
            }
        } else if (instantRotatingInPlace) {
            if (rotatingConditionStartNs == 0L) rotatingConditionStartNs = sample.timestampNs
            stationaryConditionStartNs = 0L
            val elapsedMs = (sample.timestampNs - rotatingConditionStartNs) / 1_000_000f
            if (elapsedMs >= minRotatingDurationMs) {
                motionState = NavMotionState.ROTATING_IN_PLACE
                isStationaryState = false
            }
        } else {
            stationaryConditionStartNs = 0L
            rotatingConditionStartNs = 0L
            motionState = NavMotionState.MOVING
            isStationaryState = false
        }

        // §52: log every motionState transition (not every sample -- this fires only on
        // change, so it won't flood Logcat) with the exact values that caused it. Filter
        // Logcat on tag "Gudumap:ZuptDetector" to watch this live during a walking test --
        // if STATIONARY still gets latched into during genuine walking after this fix, these
        // numbers are the ground truth for re-tuning the thresholds above, instead of guessing.
        if (motionState != previousMotionState) {
            Log.i(
                TAG,
                "motionState ${previousMotionState.name} -> ${motionState.name} | " +
                    "accMag=$accMag accVar=$accVar horizAcc=$accHoriz horizVar=$accHorizVar " +
                    "gyroMag=$gyroMag gyroVar=$gyroVar gnssSpeed=$gnssSpeed"
            )
        }

        return isStationaryState
    }

    private fun calculateVariance(history: FloatArray, count: Int): Float {
        if (count < 2) return 0f
        var sum = 0f
        for (i in 0 until count) {
            sum += history[i]
        }
        val mean = sum / count
        var varSum = 0f
        for (i in 0 until count) {
            val diff = history[i] - mean
            varSum += diff * diff
        }
        return varSum / (count - 1)
    }

    val isStationary: Boolean
        get() = isEnabled && (motionState == NavMotionState.STATIONARY)

    val isRotatingInPlace: Boolean
        get() = isEnabled && (motionState == NavMotionState.ROTATING_IN_PLACE)

    val isNavStationary: Boolean
        get() = isEnabled && (motionState != NavMotionState.MOVING)

    fun reset() {
        historyCount = 0
        historyIndex = 0
        stationaryConditionStartNs = 0L
        rotatingConditionStartNs = 0L
        isStationaryState = false
        motionState = NavMotionState.STATIONARY
    }
}
