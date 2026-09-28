package com.example.gudumap.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.gudumap.navigation.NavigationEngine
import com.example.gudumap.navigation.NavigationState
import com.example.gudumap.tracking.SessionFiles
import com.example.gudumap.tracking.SessionFrame
import com.example.gudumap.tracking.SessionRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/** §60: recording / export status shown in the Details drawer's TOOLS section. */
data class SessionUiState(
    val isRecording: Boolean = false,
    val recordedFrames: Int = 0,
    val lastSessionName: String? = null, // newest saved session file, if any
    val message: String? = null          // one-line feedback ("Saved …", "Nothing recorded yet", …)
)

/** §60: replay of a saved session. `frames` is the whole session; `index` is the frame on screen. */
data class ReplayUiState(
    val active: Boolean = false,
    val frames: List<SessionFrame> = emptyList(),
    val index: Int = 0,
    val speed: Int = 4,
    val fileName: String = ""
) {
    val current: SessionFrame? get() = frames.getOrNull(index)
    val elapsedSec: Double get() =
        if (frames.isEmpty()) 0.0 else (frames[index].wallTimeMs - frames[0].wallTimeMs) / 1000.0
    val totalSec: Double get() =
        if (frames.isEmpty()) 0.0 else (frames.last().wallTimeMs - frames[0].wallTimeMs) / 1000.0
}

/**
 * ViewModel exposing navigation state to Compose UI.
 * Connects directly to the existing NavigationEngine and pipeline:
 * Sensors -> SensorFusion -> ML -> Dead Reckoning -> EKF -> Map Matching -> NavigationEngine -> NavigationState -> ViewModel -> UI
 */
class NavigationViewModel(
    application: Application,
    private val navigationEngine: NavigationEngine = NavigationEngine(application.applicationContext)
) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(NavigationState())
    val state: StateFlow<NavigationState> = _state.asStateFlow()

    // §60: session recording / export / replay. All of this only observes `state`; none of it
    // feeds anything back into the engine or the model.
    private val filesDir: File = application.filesDir
    private val recorder = SessionRecorder()
    private val _session = MutableStateFlow(SessionUiState())
    val session: StateFlow<SessionUiState> = _session.asStateFlow()
    private val _replay = MutableStateFlow(ReplayUiState())
    val replay: StateFlow<ReplayUiState> = _replay.asStateFlow()
    private var replayJob: Job? = null

    // Direct property accessors for all required navigation & sensor metrics
    val blackoutMode: Boolean get() = state.value.blackoutMode
    val navigationMode: String get() = state.value.navigationMode
    val gnssRecovered: Boolean get() = state.value.gnssRecovered
    val recoveryDriftMeters: Double get() = state.value.recoveryDriftMeters
    val recoveryErrorPercent: Double get() = state.value.recoveryErrorPercent
    val gnssStatus: String get() = state.value.gnssStatus
    val mlStatus: String get() = state.value.mlStatus
    val ekfStatus: String get() = state.value.ekfStatus
    val mapStatus: String get() = state.value.mapStatus
    val offlineMapStatus: String get() = state.value.offlineMapStatus
    val currentRoadName: String get() = state.value.currentRoadName
    val latitude: Double get() = state.value.latitude
    val longitude: Double get() = state.value.longitude
    val speedKmh: Float get() = state.value.speedKmh
    val headingDeg: Float get() = state.value.headingDeg
    val distanceMeters: Double get() = state.value.distanceMeters
    val positionErrorMeters: Double get() = state.value.positionErrorMeters
    val driftPercentage: Double get() = state.value.driftPercentage
    val mlInferenceLatencyMs: Long get() = state.value.mlInferenceLatencyMs
    val accelerometerActive: Boolean get() = state.value.accelerometerActive
    val gyroscopeActive: Boolean get() = state.value.gyroscopeActive
    val magnetometerActive: Boolean get() = state.value.magnetometerActive
    val motionState: String get() = state.value.motionState
    val gnssNavigationMode: String get() = state.value.gnssNavigationMode
    val latestGateAction: String get() = state.value.latestGateAction
    val acceptedCount: Int get() = state.value.acceptedCount
    val clampedCount: Int get() = state.value.clampedCount
    val rejectedCount: Int get() = state.value.rejectedCount
    val blackoutMetrics: com.example.gudumap.navigation.BlackoutMetrics get() = state.value.blackoutMetrics
    val blackoutDurationSeconds: Double get() = state.value.blackoutDurationSeconds
    val gnssGroundTruthLat: Double? get() = state.value.gnssGroundTruthLat
    val gnssGroundTruthLon: Double? get() = state.value.gnssGroundTruthLon

    init {
        // Collect state emissions from NavigationEngine within viewModelScope
        viewModelScope.launch {
            navigationEngine.state.collect { latestState ->
                _state.value = latestState
                if (recorder.onState(latestState)) {
                    _session.update { it.copy(recordedFrames = recorder.frameCount) }
                }
            }
        }
        navigationEngine.start()

        viewModelScope.launch {
            val latest = withContext(Dispatchers.IO) { SessionFiles.latestCsv(filesDir) }
            _session.update { it.copy(lastSessionName = latest?.name) }
        }
    }

    // ---------------------------------------------------------------- §60: auto blackout

    fun setAutoBlackoutEnabled(enabled: Boolean) {
        navigationEngine.setAutoBlackoutEnabled(enabled)
    }

    // ---------------------------------------------------------------- §60: recording

    fun startRecording() {
        recorder.start()
        _session.update { it.copy(isRecording = true, recordedFrames = 0, message = "Recording…") }
    }

    fun stopRecordingAndSave() {
        val frames = recorder.stop()
        val startedAt = recorder.startedAtMs
        if (frames.size < 2) {
            _session.update { it.copy(isRecording = false, recordedFrames = 0, message = "Nothing recorded (need a GPS fix first)") }
            return
        }
        _session.update { it.copy(isRecording = false, message = "Saving…") }
        viewModelScope.launch {
            val saved = withContext(Dispatchers.IO) {
                try {
                    SessionFiles.newCsvFile(filesDir, startedAt).also { SessionFiles.writeCsv(it, frames) }
                } catch (e: Exception) {
                    null
                }
            }
            _session.update {
                if (saved != null) {
                    it.copy(lastSessionName = saved.name, recordedFrames = 0, message = "Saved ${frames.size} points")
                } else {
                    it.copy(message = "Couldn't save the session")
                }
            }
        }
    }

    /**
     * Writes the GPX next to the newest saved session CSV and returns both files for the share
     * sheet, or null if there's nothing saved yet. Caller turns them into content:// URIs.
     */
    suspend fun prepareLatestExport(): List<File>? = withContext(Dispatchers.IO) {
        val csv = SessionFiles.latestCsv(filesDir) ?: return@withContext null
        try {
            val gpx = SessionFiles.gpxFor(csv)
            SessionFiles.writeGpx(gpx, SessionFiles.readCsv(csv))
            listOf(csv, gpx)
        } catch (e: Exception) {
            null
        }
    }

    fun postSessionMessage(message: String?) {
        _session.update { it.copy(message = message) }
    }

    // ---------------------------------------------------------------- §60: replay

    fun startReplayOfLatest() {
        if (_replay.value.active || replayJob?.isActive == true) return
        // One job covers both loading and playback, so Stop works even while the file is loading.
        replayJob = viewModelScope.launch {
            val file = withContext(Dispatchers.IO) { SessionFiles.latestCsv(filesDir) }
            val frames: List<SessionFrame> = if (file != null) {
                withContext(Dispatchers.IO) {
                    try { SessionFiles.readCsv(file) } catch (e: Exception) { emptyList<SessionFrame>() }
                }
            } else {
                emptyList()
            }
            if (file == null || frames.size < 2) {
                _session.update { it.copy(message = "No saved session to replay yet") }
                return@launch
            }
            _replay.value = ReplayUiState(active = true, frames = frames, index = 0, speed = 4, fileName = file.name)
            while (isActive) {
                val r = _replay.value
                if (!r.active || r.index >= r.frames.size - 1) break
                val gapMs = r.frames[r.index + 1].wallTimeMs - r.frames[r.index].wallTimeMs
                // Real timing divided by the playback speed; long pauses in the recording
                // are capped so a replay never just sits still for ages.
                delay((gapMs / r.speed).coerceIn(16L, 1_500L))
                _replay.update { it.copy(index = (it.index + 1).coerceAtMost(it.frames.size - 1)) }
            }
        }
    }

    fun cycleReplaySpeed() {
        _replay.update {
            val next = when (it.speed) { 1 -> 2; 2 -> 4; 4 -> 8; else -> 1 }
            it.copy(speed = next)
        }
    }

    fun stopReplay() {
        replayJob?.cancel()
        replayJob = null
        _replay.value = ReplayUiState()
    }

    fun setBlackoutMode(enabled: Boolean) {
        navigationEngine.setBlackoutMode(enabled)
    }

    fun toggleBlackout() {
        navigationEngine.toggleBlackout()
    }

    /**
     * Re-registers for location updates if permission is now granted but we aren't already
     * listening. Call this from a permission-grant callback and from onResume -- fixes the
     * registration-timing gap where a late permission grant was never retried (PROJECT_STATUS.md
     * §13/14).
     */
    fun retryLocationUpdatesIfNeeded() {
        navigationEngine.retryLocationUpdatesIfNeeded()
    }

    /**
     * Pauses/resumes sensors and location updates for app-lifecycle background safety (merged
     * back from teammate's branch, PROJECT_STATUS.md §27) -- held back in an earlier session
     * since NavigationEngine.pause()/resume() didn't exist yet; unblocked now that they do. Not
     * yet wired to any Activity/Compose lifecycle observer on this branch -- call sites are a
     * separate step, not part of this merge.
     */
    fun pauseNavigation() {
        navigationEngine.pause()
    }

    fun resumeNavigation() {
        navigationEngine.resume()
    }

    override fun onCleared() {
        super.onCleared()
        navigationEngine.stop()
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val application = checkNotNull(this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as? Application) {
                    "Application not found in ViewModelProvider CreationExtras"
                }
                NavigationViewModel(application)
            }
        }
    }
}
