package com.example.gudumap.ui.screens

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.Crossfade
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.gudumap.navigation.BlackoutMetrics
import com.example.gudumap.navigation.NavigationState
import com.example.gudumap.ui.components.GlassCard
import com.example.gudumap.ui.components.MapView
import com.example.gudumap.ui.theme.CyanPrimary
import com.example.gudumap.ui.theme.ErrorRed
import com.example.gudumap.ui.theme.GudumapShapes
import com.example.gudumap.ui.theme.NavySurfaceVariant
import com.example.gudumap.ui.theme.OnCyanPrimary
import com.example.gudumap.ui.theme.OnErrorRed
import com.example.gudumap.ui.theme.OnStatus
import com.example.gudumap.ui.theme.OnVioletSecondary
import com.example.gudumap.ui.theme.Outline
import com.example.gudumap.ui.theme.StatusError
import com.example.gudumap.ui.theme.StatusGood
import com.example.gudumap.ui.theme.StatusWarning
import com.example.gudumap.ui.theme.TextMuted
import com.example.gudumap.ui.theme.TextPrimary
import com.example.gudumap.ui.theme.VioletSecondary
import com.example.gudumap.viewmodel.NavigationViewModel
import com.example.gudumap.viewmodel.ReplayUiState
import com.example.gudumap.viewmodel.SessionUiState
import java.util.Locale
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Plain-language confidence tier derived from the EKF's numeric uncertainty radius (§19/§23).
 * Thresholds are a first-pass judgment call for demo/UI purposes only -- not validated against
 * real measured accuracy data. Unchanged from the pre-phase-2 version.
 */
private fun confidenceLevel(radiusMeters: Double): String = when {
    radiusMeters < 5.0 -> "High"
    radiusMeters <= 15.0 -> "Medium"
    else -> "Low"
}

// Solid semantic colors from the phase-1 design system (§35), not raw hex -- swapped in here,
// same three tiers, same thresholds; only the actual color values moved to the shared token set.
private fun confidenceColor(level: String): Color = when (level) {
    "High" -> StatusGood
    "Medium" -> StatusWarning
    else -> StatusError
}

private fun confidenceCaption(level: String): String = when (level) {
    "High" -> "Position is well-established"
    "Medium" -> "Position may drift slightly"
    else -> "Recalculating -- treat position as approximate"
}

/**
 * Map-first layout (§36): MapView fills the entire screen as a Box background; every other
 * element from the pre-phase-2 scrolling Column is now a floating GlassCard overlay on top of
 * it, the same pattern most modern map-first navigation apps use. `isMapExpanded` and the old
 * true-fullscreen early-return branch are gone -- the map is always full-bleed now, so there is
 * no longer a distinct "expanded" map state to toggle. `MapView` itself is called with
 * `isExpanded = true` (its own existing fillMaxSize/no-border/no-corner-radius branch, reused
 * rather than duplicated) and `onToggleExpand = null` (its own existing floating expand button
 * simply isn't rendered, since there is nothing left to expand into -- an already-supported,
 * optional code path in `MapView.kt`, not a change to it).
 */
@Composable
fun NavigationScreen(
    navViewModel: NavigationViewModel = viewModel(factory = NavigationViewModel.Factory)
) {
    val context = LocalContext.current
    val navState by navViewModel.state.collectAsState()
    // §60: session recording/export/replay state (observes live state only; never feeds back).
    val session by navViewModel.session.collectAsState()
    val replay by navViewModel.replay.collectAsState()
    val uiScope = rememberCoroutineScope()

    var permissionGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        permissionGranted = granted
        if (granted) {
            navViewModel.retryLocationUpdatesIfNeeded()
        }
    }

    // Pause/resume sensors+location on app background/foreground (§27), and re-check permission
    // on resume in case it was granted via system Settings while backgrounded. Unchanged.
    val lifecycleOwner = LocalLifecycleOwner.current
    val latestViewModel by rememberUpdatedState(navViewModel)
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> {
                    latestViewModel.resumeNavigation()
                    permissionGranted = ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.ACCESS_FINE_LOCATION
                    ) == PackageManager.PERMISSION_GRANTED
                    latestViewModel.retryLocationUpdatesIfNeeded()
                }
                Lifecycle.Event.ON_PAUSE, Lifecycle.Event.ON_STOP -> {
                    latestViewModel.pauseNavigation()
                }
                else -> {}
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    // §56: keep the display awake while this screen is showing. The observer above pauses
    // sensors + location on ON_PAUSE/ON_STOP, so if the screen timed out mid-walk/drive the whole
    // dead-reckoning pipeline silently stopped with it -- a blackout test would just freeze
    // wherever the screen happened to sleep. Scoped to this composable's lifetime only (released
    // in onDispose), the same thing every turn-by-turn navigation app does.
    val hostView = LocalView.current
    DisposableEffect(hostView) {
        hostView.keepScreenOn = true
        onDispose { hostView.keepScreenOn = false }
    }

    var blackoutControlStage by remember { mutableStateOf(0) } // 0: GNSS AVAILABLE, 1: START GNSS BLACKOUT
    var isDetailsExpanded by remember { mutableStateOf(false) }

    // §56: the arm-then-confirm blackout flow previously stayed armed forever -- one stray tap on
    // "GNSS AVAILABLE" left a red "START GNSS BLACKOUT" button sitting there indefinitely, one
    // more accidental tap away from starting a demo-critical mode. Auto-disarm after 5s so the
    // two-tap confirm has to happen deliberately, close together. Purely UI state; the actual
    // setBlackoutMode(...) calls are unchanged.
    LaunchedEffect(blackoutControlStage) {
        if (blackoutControlStage == 1) {
            delay(5_000)
            blackoutControlStage = 0
        }
    }

    // §60: post-blackout report card. Latches the moment a blackout ends (blackoutMode true ->
    // false); the engine sets `blackoutMetrics` to the final, GPS-scored numbers in that same
    // state update. Blackouts shorter than 3 s (accidental taps) don't get a card.
    var reportMetrics by remember { mutableStateOf<BlackoutMetrics?>(null) }
    var reportAutoReason by remember { mutableStateOf("") }
    var wasInBlackout by remember { mutableStateOf(false) }
    var blackoutLatchedReason by remember { mutableStateOf("") }
    LaunchedEffect(navState.blackoutMode, navState.blackoutAutoReason) {
        if (navState.blackoutMode) {
            if (!wasInBlackout) {
                wasInBlackout = true
                blackoutLatchedReason = ""
                reportMetrics = null
            }
            if (navState.blackoutAutoReason.isNotEmpty()) blackoutLatchedReason = navState.blackoutAutoReason
        } else if (wasInBlackout) {
            wasInBlackout = false
            val finalMetrics = navState.blackoutMetrics
            if (finalMetrics.blackoutDurationSeconds >= 3.0) {
                reportMetrics = finalMetrics
                reportAutoReason = blackoutLatchedReason
            }
            blackoutLatchedReason = ""
        }
    }

    // §60: compass calibration hint -- only after heading confidence has stayed LOW/UNRELIABLE
    // for 5 s (Android reports UNRELIABLE briefly at startup before accuracy callbacks arrive).
    // Dismissing it keeps it hidden until confidence recovers and then degrades again.
    val headingPoor = navState.headingConfidence == "UNRELIABLE" || navState.headingConfidence == "LOW"
    var showCalibrationHint by remember { mutableStateOf(false) }
    var calibrationDismissed by remember { mutableStateOf(false) }
    LaunchedEffect(headingPoor) {
        if (headingPoor) {
            delay(5_000)
            if (!calibrationDismissed) showCalibrationHint = true
        } else {
            showCalibrationHint = false
            calibrationDismissed = false
        }
    }

    // §60: one-line tool feedback ("Saved 312 points", ...) clears itself after a few seconds.
    LaunchedEffect(session.message) {
        val msg = session.message
        if (msg != null && msg != "Recording…") {
            delay(4_000)
            navViewModel.postSessionMessage(null)
        }
    }

    val exportLatestSession: () -> Unit = {
        uiScope.launch {
            val files = navViewModel.prepareLatestExport()
            if (files == null) {
                navViewModel.postSessionMessage("Nothing to export yet — record a session first")
                return@launch
            }
            try {
                val authority = "${context.packageName}.fileprovider"
                val uris = ArrayList(files.map { FileProvider.getUriForFile(context, authority, it) })
                val send = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
                    type = "*/*"
                    putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
                    putExtra(Intent.EXTRA_SUBJECT, "Gudumap session ${files.first().nameWithoutExtension}")
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                context.startActivity(Intent.createChooser(send, "Export session (CSV + GPX)"))
            } catch (e: Exception) {
                navViewModel.postSessionMessage("Export failed: ${e.message}")
            }
        }
    }

    // §60: while a saved session is replaying, the map is driven by the recorded frames instead
    // of live state (live navigation keeps running underneath, untouched).
    val replayFrame = if (replay.active) replay.current else null

    Box(modifier = Modifier.fillMaxSize()) {

        // ========================================================
        // FULL-BLEED MAP BACKGROUND -- MapView.kt itself untouched.
        // Always isExpanded=true now (its existing fillMaxSize/edge-to-edge
        // branch), onToggleExpand=null (its existing floating expand button
        // simply isn't rendered -- nothing left to expand into).
        // ========================================================
        MapView(
            latitude = replayFrame?.latitude ?: navState.latitude,
            longitude = replayFrame?.longitude ?: navState.longitude,
            headingDeg = replayFrame?.headingDeg ?: navState.headingDeg,
            mapStatus = navState.mapStatus,
            offlineMapStatus = navState.offlineMapStatus,
            roadName = if (replayFrame != null) "" else navState.currentRoadName,
            blackoutMode = replayFrame?.blackout ?: navState.blackoutMode,
            // A replay frame without a naive point (outside blackout) passes 0.0, which MapView
            // already treats as "use the corrected position".
            naiveLatitude = if (replayFrame != null) (replayFrame.naiveLatitude ?: 0.0) else navState.naiveLatitude,
            naiveLongitude = if (replayFrame != null) (replayFrame.naiveLongitude ?: 0.0) else navState.naiveLongitude,
            uncertaintyRadiusMeters = replayFrame?.uncertaintyMeters ?: navState.uncertaintyRadiusMeters,
            isExpanded = true,
            onToggleExpand = null,
            // §57: reserve room for this screen's own overlays so MapView's corner controls sit
            // between them instead of underneath them. Top: TopStatusPill (12dp offset + ~40dp
            // tall). Bottom: the collapsed DetailsDrawer (12dp offset + ~42dp header) -- this
            // puts MapView's bottom-end 🎯/+/- stack level with BlackoutFab (76dp).
            overlayTopPadding = 52.dp,
            overlayBottomPadding = 64.dp,
            modifier = Modifier.fillMaxSize()
        )

        // ========================================================
        // TOP STATUS PILL -- slim, compact: plain-language state (from the
        // old StatusBanner's tri-state message/color logic, unchanged) plus
        // EKF / ML / Motion Mode readouts.
        // ========================================================
        if (replay.active) {
            ReplayPill(
                replay = replay,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .windowInsetsPadding(WindowInsets.safeDrawing)
                    .padding(top = 12.dp)
            )
        } else {
            TopStatusPill(
                navState = navState,
                isRecording = session.isRecording,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .windowInsetsPadding(WindowInsets.safeDrawing)
                    .padding(top = 12.dp)
            )
        }

        // Location permission request -- a blocking control, not a passive status readout, so
        // it floats independently of the details drawer's expand/collapse state, same as before.
        if (!permissionGranted) {
            PermissionBanner(
                onRequestPermission = {
                    permissionLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
                },
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .windowInsetsPadding(WindowInsets.safeDrawing)
                    .padding(top = 68.dp)
            )
        }

        // ========================================================
        // GNSS BLACKOUT TOGGLE -- floating action button. Bottom-start, just
        // above the collapsed details drawer; owns the bottom-left corner now
        // that MapView's duplicate "MY LOCATION" pill is gone (§57). Level
        // with MapView's bottom-end recenter/zoom stack. Same BlackoutControlButton
        // logic as before (hasGpsFix gate, two-stage arm-then-confirm),
        // restyled only.
        // ========================================================
        if (replay.active) {
            ReplayControls(
                replay = replay,
                onCycleSpeed = { navViewModel.cycleReplaySpeed() },
                onStop = { navViewModel.stopReplay() },
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .windowInsetsPadding(WindowInsets.safeDrawing)
                    .padding(horizontal = 12.dp, vertical = 12.dp)
            )
        } else {
        BlackoutFab(
            navState = navState,
            blackoutControlStage = blackoutControlStage,
            onArm = { blackoutControlStage = 1 },
            onStart = {
                navViewModel.setBlackoutMode(true)
                blackoutControlStage = 0
            },
            onEnd = {
                navViewModel.setBlackoutMode(false)
                blackoutControlStage = 0
            },
            modifier = Modifier
                .align(Alignment.BottomStart)
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(start = 16.dp, bottom = 76.dp)
        )

        // ========================================================
        // DETAILS DRAWER -- bottom-center floating GlassCard, inset from
        // the screen edges (not edge-to-edge) so MapView's own bottom-start/
        // bottom-end corner controls stay visible and reachable around it.
        // Collapsed by default (a slim header only); expands in place to
        // show the same stats the old "technical details" section + the old
        // always-visible-during-blackout position-confidence card used to
        // show, capped at a bounded max height (a scrollable column) rather
        // than ever taking the full screen.
        // ========================================================
        DetailsDrawer(
            navState = navState,
            session = session,
            expanded = isDetailsExpanded,
            onToggleExpanded = { isDetailsExpanded = !isDetailsExpanded },
            onToggleAutoBlackout = { navViewModel.setAutoBlackoutEnabled(!navState.autoBlackoutEnabled) },
            onToggleRecording = {
                if (session.isRecording) navViewModel.stopRecordingAndSave() else navViewModel.startRecording()
            },
            onExport = exportLatestSession,
            onReplay = {
                isDetailsExpanded = false
                navViewModel.startReplayOfLatest()
            },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(horizontal = 12.dp, vertical = 12.dp)
        )

        // §60: compass calibration hint -- bottom-start, clear of the right-hand zoom stack.
        if (showCalibrationHint && !isDetailsExpanded) {
            CalibrationHint(
                onDismiss = {
                    calibrationDismissed = true
                    showCalibrationHint = false
                },
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .windowInsetsPadding(WindowInsets.safeDrawing)
                    .padding(start = 16.dp, end = 64.dp, bottom = 128.dp)
            )
        }
        }

        // §60: post-blackout report card, centered over the map until dismissed.
        val report = reportMetrics
        if (report != null && !replay.active) {
            BlackoutReportCard(
                metrics = report,
                autoReason = reportAutoReason,
                onClose = { reportMetrics = null },
                modifier = Modifier
                    .align(Alignment.Center)
                    .windowInsetsPadding(WindowInsets.safeDrawing)
                    .padding(horizontal = 20.dp)
            )
        }
    }
}

/** Which of the pill's three plain-language visual states is currently active. */
private enum class PillVisualState { LIVE, BLACKOUT, RECOVERING }

/**
 * Compact floating status pill (§36, animated in §37): the old StatusBanner's plain-language
 * tri-state message/color (blackout / recovering / live tracking -- identical logic, unchanged)
 * as the leading label, plus compact EKF / ML readouts and the Motion Mode badge. Motion Mode is
 * shown only during blackout, exactly as before (§30/§24's own reasoning: outside blackout it
 * just sits at its neutral default, which would be misleading to show as if it were a real
 * classification) -- that conditional is preserved verbatim.
 *
 * Two animations, both purely cosmetic and both driven only by real state (nothing here delays
 * or gates when a real value change is reflected -- `visualState` is recomputed fresh every
 * recomposition from live `navState` fields, same as before this session):
 * - The dot+message swap between states now `Crossfade`s (300ms) instead of cutting instantly.
 * - The dot carries a small, constant, always-on alpha pulse (~1.1s cycle, 0.55-1.0 range) to
 *   read as "live" -- the SAME cycle regardless of state or actual data quality, so it never
 *   implies anything true or false about how good the current fix/estimate is.
 */
@Composable
private fun TopStatusPill(navState: NavigationState, isRecording: Boolean = false, modifier: Modifier = Modifier) {
    val visualState = when {
        navState.blackoutMode -> PillVisualState.BLACKOUT
        navState.gnssNavigationMode == "GNSS_RECOVERY" -> PillVisualState.RECOVERING
        else -> PillVisualState.LIVE
    }

    val liveDotPulse = rememberInfiniteTransition(label = "liveDotPulse")
    val pulseAlpha by liveDotPulse.animateFloat(
        initialValue = 0.55f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1100, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "liveDotPulseAlpha"
    )

    GlassCard(
        modifier = modifier.widthIn(max = 340.dp),
        shape = GudumapShapes.extraLarge
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Crossfade(
                targetState = visualState,
                animationSpec = tween(durationMillis = 300),
                label = "statusPillState",
                modifier = Modifier.weight(1f, fill = false)
            ) { state ->
                val (dotColor, message) = when (state) {
                    PillVisualState.BLACKOUT -> CyanPrimary to "Navigating without GPS"
                    PillVisualState.RECOVERING -> StatusWarning to "Reconnecting…"
                    PillVisualState.LIVE -> StatusGood to "Live Tracking"
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .alpha(pulseAlpha)
                            .background(dotColor, shape = CircleShape)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = message,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )
                }
            }
            Spacer(modifier = Modifier.width(10.dp))
            CompactStatusChip(label = "EKF", isGood = navState.ekfStatus == "ACTIVE")
            Spacer(modifier = Modifier.width(6.dp))
            CompactStatusChip(label = "ML", isGood = navState.mlStatus == "ACTIVE")
            if (navState.blackoutMode) {
                Spacer(modifier = Modifier.width(6.dp))
                MotionModeBadge(motionMode = navState.motionMode)
            }
            if (isRecording) {
                Spacer(modifier = Modifier.width(6.dp))
                Text(text = "● REC", fontSize = 10.sp, fontWeight = FontWeight.Bold, color = StatusError)
            }
        }
    }
}

@Composable
private fun CompactStatusChip(label: String, isGood: Boolean) {
    val color = if (isGood) StatusGood else StatusError
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(6.dp)
                .background(color, shape = CircleShape)
        )
        Spacer(modifier = Modifier.width(3.dp))
        Text(text = label, fontSize = 10.sp, fontWeight = FontWeight.Bold, color = TextMuted)
    }
}

/**
 * Motion Mode indicator (§30/§35 restyle) -- the pedestrian-safety fallback (§24/§25) badge.
 * Same two states/colors-with-meaning as before (blue/car for VEHICLE_MODE, amber/walking for
 * CONSERVATIVE_MODE), now drawn from the phase-1 token set instead of raw hex.
 */
@Composable
private fun MotionModeBadge(motionMode: String, modifier: Modifier = Modifier) {
    val isConservative = motionMode == "CONSERVATIVE_MODE"
    val icon = if (isConservative) "🚶" else "🚗"
    val label = if (isConservative) "Conservative" else "Vehicle"
    val color = if (isConservative) StatusWarning else VioletSecondary
    val onColor = if (isConservative) OnStatus else OnVioletSecondary

    Box(
        modifier = modifier
            .background(color, shape = RoundedCornerShape(20.dp))
            .padding(horizontal = 8.dp, vertical = 3.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(text = icon, fontSize = 10.sp)
            Spacer(modifier = Modifier.width(3.dp))
            Text(text = label, fontSize = 10.sp, fontWeight = FontWeight.Bold, color = onColor)
        }
    }
}

/**
 * Blocking location-permission prompt, floated as its own compact GlassCard directly below the
 * status pill. Same trigger condition and same `permissionLauncher.launch(...)` call as before,
 * just no longer a full-width Column child.
 */
@Composable
private fun PermissionBanner(onRequestPermission: () -> Unit, modifier: Modifier = Modifier) {
    GlassCard(
        modifier = modifier.widthIn(max = 340.dp),
        shape = GudumapShapes.large
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = "Location permission needed",
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                color = TextPrimary
            )
            Spacer(modifier = Modifier.height(8.dp))
            Button(
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = CyanPrimary, contentColor = OnCyanPrimary),
                shape = GudumapShapes.medium,
                onClick = onRequestPermission
            ) {
                Text(text = "GRANT LOCATION PERMISSION", fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

/** Small holder for BlackoutFab's per-state visuals, so the `when` block reads as one table. */
private data class FabSpec(
    val bg: Color,
    val onBg: Color,
    val label: String,
    val enabled: Boolean,
    val onClick: () -> Unit
)

/**
 * The GNSS-blackout trigger/status control, restyled as a floating action button. Every branch,
 * every condition, and every label is byte-for-byte identical to the pre-phase-2
 * `BlackoutControlButton` -- the `hasGpsFix`-gated disabled state (Fix 2, §11) and the two-stage
 * arm-then-confirm flow to start (prevents an accidental tap from starting a demo-critical mode)
 * are non-negotiable per this session's own instructions, so only the visual container (a
 * squircle pill sized to its label instead of a full-width Material3 `Button`) and its colors
 * (phase-1 tokens instead of raw hex, same semantic mapping) changed.
 */
@Composable
private fun BlackoutFab(
    navState: NavigationState,
    blackoutControlStage: Int,
    onArm: () -> Unit,
    onStart: () -> Unit,
    onEnd: () -> Unit,
    modifier: Modifier = Modifier
) {
    val spec = when {
        !navState.hasGpsFix ->
            FabSpec(Outline, TextMuted, "WAITING FOR GPS FIX...", false, {})
        navState.blackoutMode && navState.blackoutAutoReason == "GPS_LOSS" ->
            FabSpec(ErrorRed, OnErrorRed, "AUTO: GPS LOST (TAP TO END)", true, onEnd)
        navState.blackoutMode && navState.blackoutAutoReason == "NETWORK_LOSS" ->
            FabSpec(ErrorRed, OnErrorRed, "AUTO: OFFLINE (TAP TO END)", true, onEnd)
        navState.blackoutMode ->
            FabSpec(ErrorRed, OnErrorRed, "GNSS BLACKOUT ACTIVE (TAP TO END)", true, onEnd)
        navState.gnssNavigationMode == "GNSS_RECOVERY" ->
            FabSpec(StatusWarning, OnStatus, "RECOVERING GNSS...", false, {})
        blackoutControlStage == 1 ->
            FabSpec(ErrorRed, OnErrorRed, "START GNSS BLACKOUT", true, onStart)
        else ->
            FabSpec(
                StatusGood, OnStatus,
                if (navState.autoBlackoutEnabled) "GNSS AVAILABLE · AUTO ON" else "GNSS AVAILABLE",
                true, onArm
            )
    }

    Button(
        modifier = modifier,
        colors = ButtonDefaults.buttonColors(
            containerColor = spec.bg,
            contentColor = spec.onBg,
            disabledContainerColor = spec.bg,
            disabledContentColor = spec.onBg
        ),
        shape = GudumapShapes.extraLarge,
        enabled = spec.enabled,
        onClick = spec.onClick
    ) {
        Text(text = spec.label, fontSize = 11.sp, fontWeight = FontWeight.Bold)
    }
}

/**
 * Bottom-center floating details drawer (§36). Collapsed by default -- a slim header row (label
 * + expand/collapse chevron) only, never taking full screen height. Expanded content is the same
 * set of stats the pre-phase-2 "technical details" section + the always-during-blackout
 * position-confidence card used to show (blackout metrics, navigation status, position,
 * navigation metrics, sensor status, position confidence), unchanged in fields/logic and capped
 * at a bounded max height via an inner scrollable Column so a long expanded panel still can't
 * cover the whole screen.
 */
@Composable
private fun DetailsDrawer(
    navState: NavigationState,
    session: SessionUiState,
    expanded: Boolean,
    onToggleExpanded: () -> Unit,
    onToggleAutoBlackout: () -> Unit,
    onToggleRecording: () -> Unit,
    onExport: () -> Unit,
    onReplay: () -> Unit,
    modifier: Modifier = Modifier
) {
    // §56/§59: more opaque than the default glass (0.90 -> 0.76 vs 0.72 -> 0.58) plus a faint
    // white frost sheen, so it reads as frosted glass: the map is still dimly visible through
    // it, but road lines and labels no longer compete with the stat text on top.
    GlassCard(
        modifier = modifier.fillMaxWidth(),
        shape = GudumapShapes.large,
        opacity = 0.90f,
        frost = 0.07f
    ) {
        Column {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable(onClick = onToggleExpanded)
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "Details",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )
                    // §56: while collapsed during a blackout, surface the two numbers people
                    // actually open the drawer for, so a demo doesn't need the drawer expanded
                    // (covering the map/trails) just to read them.
                    if (!expanded && navState.blackoutMode) {
                        Spacer(modifier = Modifier.width(10.dp))
                        Text(
                            text = String.format(
                                Locale.US,
                                "DR %.1f m · ±%.1f m",
                                navState.blackoutMetrics.drDistance,
                                navState.uncertaintyRadiusMeters
                            ),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = CyanPrimary
                        )
                    }
                    if (session.isRecording) {
                        Spacer(modifier = Modifier.width(10.dp))
                        Text(
                            text = "● REC ${session.recordedFrames}",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            color = StatusError
                        )
                    }
                }
                Text(
                    text = if (expanded) "▲ Hide" else "▼ Show",
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = TextMuted
                )
            }

            AnimatedVisibility(
                visible = expanded,
                enter = expandVertically() + fadeIn(),
                exit = shrinkVertically() + fadeOut()
            ) {
                Column(
                    modifier = Modifier
                        .heightIn(max = 420.dp)
                        .verticalScroll(rememberScrollState())
                        .padding(horizontal = 16.dp)
                        .padding(bottom = 16.dp)
                ) {
                    // §60: tools -- automatic GPS-loss detection, session recording, export, replay.
                    DetailSectionLabel("TOOLS")
                    Spacer(modifier = Modifier.height(8.dp))
                    ToolButton(
                        label = if (navState.autoBlackoutEnabled) "Auto-detect GPS loss: ON" else "Auto-detect GPS loss: OFF",
                        active = navState.autoBlackoutEnabled,
                        onClick = onToggleAutoBlackout,
                        modifier = Modifier.fillMaxWidth()
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        ToolButton(
                            label = if (session.isRecording) "■ Stop & save" else "● Record",
                            active = session.isRecording,
                            onClick = onToggleRecording,
                            modifier = Modifier.weight(1f)
                        )
                        ToolButton(
                            label = "Export",
                            enabled = session.lastSessionName != null,
                            onClick = onExport,
                            modifier = Modifier.weight(1f)
                        )
                        ToolButton(
                            label = "▶ Replay",
                            enabled = session.lastSessionName != null,
                            onClick = onReplay,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    val toolNote = session.message ?: session.lastSessionName?.let { "Last saved: $it" }
                    if (toolNote != null) {
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(text = toolNote, fontSize = 11.sp, color = TextMuted)
                    }
                    Spacer(modifier = Modifier.height(16.dp))

                    // Position confidence -- only meaningful once blackout has introduced real
                    // drift uncertainty, same conditional as before.
                    if (navState.blackoutMode) {
                        PlainConfidenceCard(uncertaintyRadiusMeters = navState.uncertaintyRadiusMeters)
                        Spacer(modifier = Modifier.height(16.dp))
                    }

                    // §56: outside a blackout these tiles hold the LAST blackout's final numbers
                    // (storedBlackoutMetrics), not live values -- label them as such.
                    DetailSectionLabel(if (navState.blackoutMode) "BLACKOUT METRICS · LIVE" else "LAST BLACKOUT")
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        MetricTile(
                            title = "DR Distance",
                            value = String.format(Locale.US, "%.1f m", navState.blackoutMetrics.drDistance),
                            modifier = Modifier.weight(1f)
                        )
                        MetricTile(
                            title = "Max Error",
                            value = String.format(Locale.US, "%.1f m", navState.blackoutMetrics.maximumPositionErrorMeters),
                            modifier = Modifier.weight(1f)
                        )
                        // §56: was a second "ML Latency" tile, an exact duplicate of "ML Inference"
                        // under NAVIGATION METRICS below. Blackout duration is the more useful
                        // number here (it gives DR Distance / Max Error their context).
                        MetricTile(
                            title = "Duration",
                            value = formatDuration(navState.blackoutMetrics.blackoutDurationSeconds),
                            modifier = Modifier.weight(1f)
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        MetricTile(
                            title = "Confidence Radius",
                            value = String.format(Locale.US, "±%.1f m", navState.uncertaintyRadiusMeters),
                            modifier = Modifier.weight(1f)
                        )
                        HeadingConfidenceTile(
                            confidence = navState.headingConfidence,
                            modifier = Modifier.weight(1f)
                        )
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    DetailSectionLabel("NAVIGATION STATUS")
                    Spacer(modifier = Modifier.height(8.dp))

                    val gnssDisplay = when {
                        navState.blackoutMode -> "BLACKOUT"
                        navState.gnssNavigationMode == "GNSS_RECOVERY" -> "RECOVERING"
                        else -> navState.gnssStatus
                    }
                    val navDisplay = if (navState.blackoutMode) "DR" else "GNSS"
                    val gateDisplay = "${navState.latestGateAction} (A:${navState.acceptedCount} C:${navState.clampedCount} R:${navState.rejectedCount})"

                    StatusRow(label = "GNSS", value = gnssDisplay, isGood = gnssDisplay == "AVAILABLE")
                    StatusRow(label = "Navigation", value = navDisplay, isGood = navDisplay == "GNSS")
                    StatusRow(label = "Motion", value = navState.motionState, isGood = true)
                    StatusRow(label = "ML Gate", value = gateDisplay, isGood = navState.latestGateAction == "ACCEPTED")
                    StatusRow(label = "ML", value = navState.mlStatus, isGood = navState.mlStatus == "ACTIVE")
                    StatusRow(label = "EKF", value = navState.ekfStatus, isGood = navState.ekfStatus == "ACTIVE")
                    StatusRow(label = "MAP", value = navState.mapStatus, isGood = true)
                    StatusRow(label = "OFFLINE MAP", value = navState.offlineMapStatus, isGood = navState.offlineMapStatus == "AVAILABLE")
                    StatusRow(label = "Internet", value = if (navState.isInternetAvailable) "AVAILABLE" else "UNAVAILABLE", isGood = navState.isInternetAvailable)

                    if (navState.currentRoadName.isNotBlank()) {
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            text = "Road: ${navState.currentRoadName}",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = CyanPrimary
                        )
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    DetailSectionLabel("POSITION")
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column {
                            Text(text = "Latitude", fontSize = 11.sp, color = TextMuted)
                            Text(
                                text = String.format(Locale.US, "%.6f", navState.latitude),
                                fontSize = 17.sp,
                                fontWeight = FontWeight.Bold,
                                color = TextPrimary
                            )
                        }
                        Column {
                            Text(text = "Longitude", fontSize = 11.sp, color = TextMuted)
                            Text(
                                text = String.format(Locale.US, "%.6f", navState.longitude),
                                fontSize = 17.sp,
                                fontWeight = FontWeight.Bold,
                                color = TextPrimary
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    DetailSectionLabel("NAVIGATION METRICS")
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        MetricTile(
                            title = "Speed",
                            value = String.format(Locale.US, "%.1f km/h", navState.speedKmh),
                            modifier = Modifier.weight(1f)
                        )
                        MetricTile(
                            title = "Heading",
                            value = String.format(Locale.US, "%.0f°", navState.headingDeg),
                            modifier = Modifier.weight(1f)
                        )
                        MetricTile(
                            title = "Distance",
                            value = String.format(Locale.US, "%.1f m", navState.distanceMeters),
                            modifier = Modifier.weight(1f)
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        MetricTile(
                            title = "DR Error",
                            value = String.format(Locale.US, "%.2f m", navState.positionErrorMeters),
                            modifier = Modifier.weight(1f)
                        )
                        MetricTile(
                            title = "Drift",
                            value = String.format(Locale.US, "%.2f %%", navState.driftPercentage),
                            modifier = Modifier.weight(1f)
                        )
                        MetricTile(
                            title = "ML Inference",
                            value = "${navState.mlInferenceLatencyMs} ms",
                            modifier = Modifier.weight(1f)
                        )
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    DetailSectionLabel("SENSOR STATUS")
                    Spacer(modifier = Modifier.height(8.dp))
                    SensorStatusRow(name = "Accelerometer", isActive = navState.accelerometerActive)
                    SensorStatusRow(name = "Gyroscope", isActive = navState.gyroscopeActive)
                    SensorStatusRow(name = "Magnetometer", isActive = navState.magnetometerActive)
                }
            }
        }
    }
}

/** §60: small pill button used in the TOOLS section. `active` = highlighted (on/recording). */
@Composable
private fun ToolButton(
    label: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    active: Boolean = false,
    enabled: Boolean = true
) {
    val bg = when {
        !enabled -> NavySurfaceVariant.copy(alpha = 0.5f)
        active -> CyanPrimary
        else -> NavySurfaceVariant
    }
    val fg = when {
        !enabled -> TextMuted.copy(alpha = 0.6f)
        active -> OnCyanPrimary
        else -> TextPrimary
    }
    Box(
        modifier = modifier
            .background(bg, shape = GudumapShapes.small)
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 10.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(text = label, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = fg)
    }
}

/**
 * §60: shown when a blackout ends. Every number comes straight from the engine's final
 * [BlackoutMetrics] (computed at the moment GPS was restored) -- nothing is recomputed here.
 */
@Composable
private fun BlackoutReportCard(
    metrics: BlackoutMetrics,
    autoReason: String, // "GPS_LOSS" / "NETWORK_LOSS" / "" (manual)
    onClose: () -> Unit,
    modifier: Modifier = Modifier
) {
    GlassCard(
        modifier = modifier.widthIn(max = 380.dp),
        shape = GudumapShapes.large,
        opacity = 0.94f,
        frost = 0.07f
    ) {
        Column(modifier = Modifier.padding(18.dp)) {
            Text(text = "Blackout report", fontSize = 17.sp, fontWeight = FontWeight.ExtraBold, color = TextPrimary)
            Text(
                text = when (autoReason) {
                    "GPS_LOSS" -> "GPS was lost — detected automatically"
                    "NETWORK_LOSS" -> "Started automatically when internet dropped"
                    else -> "Simulated GPS blackout (started manually)"
                },
                fontSize = 12.sp,
                color = TextMuted
            )
            Spacer(modifier = Modifier.height(14.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = "Error when GPS returned", fontSize = 11.sp, color = TextMuted)
                    Text(
                        text = String.format(Locale.US, "%.1f m", metrics.positionErrorMeters),
                        fontSize = 28.sp,
                        fontWeight = FontWeight.ExtraBold,
                        color = CyanPrimary
                    )
                }
                Column(horizontalAlignment = Alignment.End) {
                    Text(text = "Drift", fontSize = 11.sp, color = TextMuted)
                    Text(
                        text = if (metrics.drDistance >= 1.0) String.format(Locale.US, "%.1f %%", metrics.driftPercentage) else "—",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricTile(title = "Duration", value = formatDuration(metrics.blackoutDurationSeconds), modifier = Modifier.weight(1f))
                MetricTile(title = "DR distance", value = String.format(Locale.US, "%.1f m", metrics.drDistance), modifier = Modifier.weight(1f))
                MetricTile(title = "Max error", value = String.format(Locale.US, "%.1f m", metrics.maximumPositionErrorMeters), modifier = Modifier.weight(1f))
            }
            Spacer(modifier = Modifier.height(8.dp))
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricTile(title = "GPS start→end", value = String.format(Locale.US, "%.1f m", metrics.gnssReferenceDistance), modifier = Modifier.weight(1f))
                MetricTile(title = "Stationary", value = formatDuration(metrics.stationaryDuration), modifier = Modifier.weight(1f))
                MetricTile(
                    title = "ML gate A/C/R",
                    value = "${metrics.numberOfAcceptedMLPredictions}/${metrics.numberOfClampedMLPredictions}/${metrics.numberOfRejectedMLPredictions}",
                    modifier = Modifier.weight(1f)
                )
            }

            Spacer(modifier = Modifier.height(10.dp))
            Text(
                text = "Error = distance between the app's dead-reckoned position and the real GPS fix when navigation resumed. " +
                    "GPS start→end is straight-line, not path length.",
                fontSize = 10.sp,
                color = TextMuted
            )
            Spacer(modifier = Modifier.height(12.dp))
            Button(
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = CyanPrimary, contentColor = OnCyanPrimary),
                shape = GudumapShapes.medium,
                onClick = onClose
            ) {
                Text(text = "CLOSE", fontSize = 12.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

/** §60: nudges a figure-8 compass calibration when heading confidence stays poor. */
@Composable
private fun CalibrationHint(onDismiss: () -> Unit, modifier: Modifier = Modifier) {
    GlassCard(modifier = modifier, shape = GudumapShapes.medium, opacity = 0.90f, frost = 0.07f) {
        Row(
            modifier = Modifier.padding(start = 14.dp, end = 6.dp, top = 10.dp, bottom = 10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(text = "∞", fontSize = 22.sp, fontWeight = FontWeight.Bold, color = StatusWarning)
            Spacer(modifier = Modifier.width(10.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(text = "Compass needs calibrating", fontSize = 13.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                Text(
                    text = "Move your phone in a figure-8 a few times, away from metal and magnets.",
                    fontSize = 11.sp,
                    color = TextMuted
                )
            }
            Box(
                modifier = Modifier
                    .clickable(onClick = onDismiss)
                    .padding(10.dp)
            ) {
                Text(text = "✕", fontSize = 14.sp, color = TextMuted)
            }
        }
    }
}

/** §60: replaces the status pill while a saved session is replaying. */
@Composable
private fun ReplayPill(replay: ReplayUiState, modifier: Modifier = Modifier) {
    GlassCard(modifier = modifier.widthIn(max = 340.dp), shape = GudumapShapes.extraLarge) {
        Row(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(text = "▶", fontSize = 12.sp, color = VioletSecondary)
            Spacer(modifier = Modifier.width(8.dp))
            Text(text = "Replaying saved session", fontSize = 13.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
            if (replay.current?.blackout == true) {
                Spacer(modifier = Modifier.width(8.dp))
                Text(text = "BLACKOUT", fontSize = 10.sp, fontWeight = FontWeight.Bold, color = StatusError)
            }
        }
    }
}

/** §60: replay progress + speed + stop, in place of the blackout button and details drawer. */
@Composable
private fun ReplayControls(
    replay: ReplayUiState,
    onCycleSpeed: () -> Unit,
    onStop: () -> Unit,
    modifier: Modifier = Modifier
) {
    val fraction = if (replay.totalSec > 0.0) (replay.elapsedSec / replay.totalSec).toFloat().coerceIn(0f, 1f) else 0f
    GlassCard(modifier = modifier.fillMaxWidth(), shape = GudumapShapes.large, opacity = 0.90f, frost = 0.07f) {
        Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "${formatDuration(replay.elapsedSec)} / ${formatDuration(replay.totalSec)}",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary,
                    modifier = Modifier.weight(1f)
                )
                ToolButton(label = "${replay.speed}×", onClick = onCycleSpeed)
                Spacer(modifier = Modifier.width(8.dp))
                ToolButton(label = "■ Stop", onClick = onStop, active = true)
            }
            Spacer(modifier = Modifier.height(10.dp))
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(4.dp)
                    .background(NavySurfaceVariant, shape = RoundedCornerShape(2.dp))
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(fraction)
                        .height(4.dp)
                        .background(CyanPrimary, shape = RoundedCornerShape(2.dp))
                )
            }
            Spacer(modifier = Modifier.height(6.dp))
            Text(text = replay.fileName, fontSize = 10.sp, color = TextMuted)
        }
    }
}

/** m:ss for blackout durations (e.g. 49.5 s -> "0:49", 125 s -> "2:05"). */
private fun formatDuration(seconds: Double): String {
    val total = seconds.coerceAtLeast(0.0).toLong()
    return String.format(Locale.US, "%d:%02d", total / 60, total % 60)
}

@Composable
private fun DetailSectionLabel(text: String) {
    Text(
        text = text,
        fontSize = 11.sp,
        fontWeight = FontWeight.Bold,
        color = TextMuted,
        letterSpacing = 1.sp
    )
}

/**
 * Plain-language confidence tier, shown only during blackout (unchanged condition, now placed
 * inside the details drawer instead of always-visible in the old scrolling Column). Solid
 * indicator + one-line caption (§23 redesign), restyled onto the phase-1 tokens.
 */
@Composable
private fun PlainConfidenceCard(uncertaintyRadiusMeters: Double, modifier: Modifier = Modifier) {
    val level = confidenceLevel(uncertaintyRadiusMeters)
    val color = confidenceColor(level)
    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(NavySurfaceVariant, shape = GudumapShapes.medium)
            .padding(vertical = 12.dp, horizontal = 14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(10.dp)
                .background(color, shape = CircleShape)
        )
        Spacer(modifier = Modifier.width(10.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = "Position confidence",
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold,
                color = TextPrimary
            )
            Text(
                text = confidenceCaption(level),
                fontSize = 12.sp,
                color = TextMuted
            )
        }
        Spacer(modifier = Modifier.width(10.dp))
        Box(
            modifier = Modifier
                .background(color, shape = RoundedCornerShape(8.dp))
                .padding(horizontal = 14.dp, vertical = 8.dp)
        ) {
            Text(
                text = level,
                fontSize = 16.sp,
                fontWeight = FontWeight.ExtraBold,
                color = OnStatus
            )
        }
    }
}

@Composable
private fun StatusRow(label: String, value: String, isGood: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = label, fontSize = 13.sp, color = TextMuted)
        Box(
            modifier = Modifier
                .background(
                    if (isGood) StatusGood.copy(alpha = 0.18f) else StatusError.copy(alpha = 0.18f),
                    shape = RoundedCornerShape(6.dp)
                )
                .padding(horizontal = 8.dp, vertical = 2.dp)
        ) {
            Text(
                text = value,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = if (isGood) StatusGood else StatusError
            )
        }
    }
}

@Composable
private fun SensorStatusRow(name: String, isActive: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = name, fontSize = 13.sp, color = TextMuted)
        Box(
            modifier = Modifier
                .background(
                    if (isActive) StatusGood.copy(alpha = 0.18f) else StatusError.copy(alpha = 0.18f),
                    shape = RoundedCornerShape(6.dp)
                )
                .padding(horizontal = 8.dp, vertical = 2.dp)
        ) {
            Text(
                text = if (isActive) "ACTIVE" else "INACTIVE",
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = if (isActive) StatusGood else StatusError
            )
        }
    }
}

@Composable
private fun MetricTile(title: String, value: String, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .background(NavySurfaceVariant, shape = GudumapShapes.small)
            .padding(10.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(text = title, fontSize = 10.sp, fontWeight = FontWeight.SemiBold, color = TextMuted)
        Spacer(modifier = Modifier.height(4.dp))
        Text(text = value, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
    }
}

/**
 * Surfaces Android's own magnetometer/rotation-vector reliability signal for heading, so a
 * degraded compass (common inside a vehicle chassis or a tunnel/parking-garage's rebar) is
 * visible rather than silently assumed fine. Unchanged logic, restyled onto phase-1 tokens.
 */
@Composable
private fun HeadingConfidenceTile(confidence: String, modifier: Modifier = Modifier) {
    val color = when (confidence) {
        "HIGH" -> StatusGood
        "MEDIUM" -> StatusWarning
        else -> StatusError // LOW or UNRELIABLE
    }
    Column(
        modifier = modifier
            .background(NavySurfaceVariant, shape = GudumapShapes.small)
            .padding(10.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(text = "Heading Conf.", fontSize = 10.sp, fontWeight = FontWeight.SemiBold, color = TextMuted)
        Spacer(modifier = Modifier.height(4.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .background(color, shape = CircleShape)
            )
            Spacer(modifier = Modifier.width(4.dp))
            Text(text = confidence, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = color)
        }
    }
}
