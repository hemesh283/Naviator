package com.example.gudumap.ui.components

import android.content.Context
import android.util.Log
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.toBitmap
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.example.gudumap.R
import com.example.gudumap.map.OfflineMapManager
import kotlin.math.cos
import kotlin.math.sin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.maplibre.android.MapLibre
import org.maplibre.android.camera.CameraPosition
import org.maplibre.android.camera.CameraUpdateFactory
import org.maplibre.android.geometry.LatLng
import org.maplibre.android.maps.MapLibreMap
import org.maplibre.android.maps.MapLibreMapOptions
import org.maplibre.android.maps.MapView as MapLibreMapView
import org.maplibre.android.maps.Style
import org.maplibre.android.style.layers.FillLayer
import org.maplibre.android.style.layers.LineLayer
import org.maplibre.android.style.layers.PropertyFactory
import org.maplibre.android.style.layers.SymbolLayer
import org.maplibre.android.style.sources.GeoJsonSource

private const val TAG = "Gudumap:MapView"
private const val EARTH_RADIUS_METERS = 6371000.0
private const val MAX_TRAIL_POINTS = 2000

private const val SRC_VEHICLE = "vehicle_marker_source"
private const val LAYER_VEHICLE = "vehicle_marker_layer"
private const val ICON_VEHICLE = "vehicle_marker_icon"
private const val SRC_CORRECTED_TRAIL = "corrected_trail_source"
private const val LAYER_CORRECTED_TRAIL = "corrected_trail_layer"
private const val SRC_NAIVE_TRAIL = "naive_trail_source"
private const val LAYER_NAIVE_TRAIL = "naive_trail_layer"
private const val SRC_UNCERTAINTY = "uncertainty_circle_source"
private const val LAYER_UNCERTAINTY_FILL = "uncertainty_circle_fill"
private const val LAYER_UNCERTAINTY_LINE = "uncertainty_circle_line"
private const val SRC_PINPOINT = "pinpoint_ring_source"
private const val LAYER_PINPOINT_FILL = "pinpoint_ring_fill"
private const val LAYER_PINPOINT_LINE = "pinpoint_ring_line"

private val EMPTY_FEATURE_COLLECTION = """{"type":"FeatureCollection","features":[]}"""

/**
 * §42 migration: MapLibre Native replacing osmdroid, for real vector-tile detail (roads, POIs,
 * buildings, landcover) matching the level of detail in the Organic Maps reference this phase
 * was scoped against -- but styled in this app's own dark navy/cyan/violet palette (Phase 1's
 * Color.kt), not Organic Maps' literal light colors, per this phase's own explicit design call.
 * See `assets/maps/coimbatore/style_template.json` for the actual style, and this session's
 * PROJECT_STATUS.md entry for the research trail behind every architectural choice below
 * (MBTiles over PMTiles, `mbtiles://<abs-path>` over a custom LocalTilesSource, `fromUri("file://")`
 * over `fromJson()`).
 *
 * The Compose function signature below is byte-for-byte unchanged from the pre-migration
 * (osmdroid) version -- re-verified directly against `NavigationScreen.kt`'s actual call site
 * before writing this file, not assumed from memory -- so `NavigationScreen.kt` needed zero
 * changes for this migration.
 *
 * Every feature the pre-migration version had is preserved: the live heading-rotated vehicle
 * marker (still smoothed via the Phase 3 precision-safe glide, unchanged math, only the render
 * target changed from an osmdroid `Marker` to a MapLibre `SymbolLayer`), the dual naive-vs-
 * corrected trail overlay during blackout (still two separate line sources, same show/hide-on-
 * blackout logic), the uncertainty-radius circle (still smoothed via Phase 3's `animateFloatAsState`,
 * now rendered as a real geodesic polygon computed by this file instead of osmdroid's
 * `Polygon.pointsAsCircle()`, since MapLibre has no built-in equivalent), and the offline-only
 * constraint -- nothing in this file or the style JSON references a network URL; the vector
 * source is `mbtiles://<local path>` (no doubled `file://` -- see OfflineMapManager's
 * `getMbtilesSourceUri()` doc for why that form is rejected), and the style itself is loaded via a local
 * `file://` URI resolved by `OfflineMapManager`.
 *
 * `isDarkMode` is kept in the signature for compatibility (no call site passes it explicitly)
 * but is now a no-op: the old raster tiles needed a runtime `ColorMatrixColorFilter` to fake a
 * dark appearance, since real OSM raster tiles are always light. The new vector style is already
 * permanently dark navy by design (the whole point of this migration's color decision), so there
 * is nothing left for a separate dark-mode filter to do.
 */
@Composable
fun MapView(
    latitude: Double,
    longitude: Double,
    headingDeg: Float = 0f,
    mapStatus: String = "OFFLINE",
    offlineMapStatus: String = "AVAILABLE",
    roadName: String = "",
    blackoutMode: Boolean = false,
    naiveLatitude: Double = latitude,
    naiveLongitude: Double = longitude,
    uncertaintyRadiusMeters: Double = 0.0,
    isExpanded: Boolean = false,
    isDarkMode: Boolean = false,
    onToggleExpand: (() -> Unit)? = null,
    // §57: space the host screen's own floating overlays occupy at the top/bottom of the map
    // (status pill above, details drawer below). MapView's own corner controls are laid out
    // inside the remaining area so the two sets of overlays never stack on top of each other.
    // Defaults of 0.dp keep the embedded (non-full-screen) card layout exactly as before.
    overlayTopPadding: Dp = 0.dp,
    overlayBottomPadding: Dp = 0.dp,
    // §61: lets the host hide the recenter/zoom stack while one of its own panels (expanded
    // details drawer, blackout report card) covers that corner -- otherwise the buttons show
    // through the translucent glass and overlap the panel's content.
    showControls: Boolean = true,
    modifier: Modifier = Modifier
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val mapScope = rememberCoroutineScope()

    val correctedTrail = remember { mutableListOf<LatLng>() }
    val naiveTrail = remember { mutableListOf<LatLng>() }
    val wasBlackout = remember { mutableStateOf(false) }

    var mapLibreMapRef by remember { mutableStateOf<MapLibreMap?>(null) }
    var mapLibreViewRef by remember { mutableStateOf<MapLibreMapView?>(null) }
    var styleReady by remember { mutableStateOf(false) }

    // §50: free-pan support. The camera used to be force-recentered on every single
    // recomposition (see the old unconditional `map.moveCamera(...)` at the bottom of the
    // `update` lambda below), which fired on every location/heading update -- multiple times
    // a second during navigation -- so any manual pan gesture was immediately snapped back
    // before the user could see anything away from the vehicle. `isFollowingUser` gates that
    // auto-recenter: true (the default) keeps the old locked-follow behavior; a user-initiated
    // gesture (detected via OnCameraMoveStartedListener below) flips it to false so the map
    // stays wherever the user panned it to, until they tap MY LOCATION or the recenter target
    // button, which both flip it back to true.
    var isFollowingUser by remember { mutableStateOf(true) }

    // Smooth vehicle-marker glide between real position updates (Phase 3, §37) -- unchanged
    // math from the pre-migration version. Only the drawn marker icon's position is
    // interpolated; every other reader of latitude/longitude below (trails, the uncertainty
    // circle, the pinpoint ring, map recentering) uses the raw real Double values directly.
    var markerFromLat by remember { mutableStateOf(latitude) }
    var markerFromLon by remember { mutableStateOf(longitude) }
    var markerToLat by remember { mutableStateOf(latitude) }
    var markerToLon by remember { mutableStateOf(longitude) }
    val markerProgress = remember { Animatable(1f) }

    LaunchedEffect(latitude, longitude) {
        val p = markerProgress.value.toDouble()
        markerFromLat += (markerToLat - markerFromLat) * p
        markerFromLon += (markerToLon - markerFromLon) * p
        markerToLat = latitude
        markerToLon = longitude
        markerProgress.snapTo(0f)
        markerProgress.animateTo(1f, animationSpec = tween(durationMillis = 300, easing = LinearEasing))
    }

    val displayedMarkerLat = markerFromLat + (markerToLat - markerFromLat) * markerProgress.value.toDouble()
    val displayedMarkerLon = markerFromLon + (markerToLon - markerFromLon) * markerProgress.value.toDouble()

    // Smooth uncertainty-radius growth/shrink (Phase 3, §37) -- unchanged. Only the drawn
    // radius is smoothed; visibility itself still gates on the real, unsmoothed value below.
    val animatedUncertaintyRadius by animateFloatAsState(
        targetValue = uncertaintyRadiusMeters.toFloat(),
        animationSpec = tween(durationMillis = 400),
        label = "uncertaintyRadius"
    )

    // Forward Activity/Fragment lifecycle events to the classic MapLibre View, same pattern
    // NavigationScreen.kt already uses elsewhere in this app for navViewModel pause/resume.
    DisposableEffect(lifecycleOwner, mapLibreViewRef) {
        val view = mapLibreViewRef
        if (view == null) {
            onDispose {}
        } else {
            val observer = LifecycleEventObserver { _, event ->
                when (event) {
                    Lifecycle.Event.ON_START -> view.onStart()
                    Lifecycle.Event.ON_RESUME -> view.onResume()
                    Lifecycle.Event.ON_PAUSE -> view.onPause()
                    Lifecycle.Event.ON_STOP -> view.onStop()
                    Lifecycle.Event.ON_DESTROY -> view.onDestroy()
                    else -> {}
                }
            }
            lifecycleOwner.lifecycle.addObserver(observer)
            onDispose {
                lifecycleOwner.lifecycle.removeObserver(observer)
            }
        }
    }

    Box(
        modifier = modifier
            .fillMaxWidth()
            // §58: full-screen gets no clip/border at all (it was a 0dp-radius clip + 0dp border,
            // i.e. an extra clipping graphics layer over the map for no visible effect). Only the
            // embedded card layout rounds and outlines the map.
            .then(
                if (isExpanded) {
                    Modifier.fillMaxSize()
                } else {
                    Modifier
                        .height(340.dp)
                        .clip(RoundedCornerShape(20.dp))
                        .border(1.dp, Color(0xFF334155), RoundedCornerShape(20.dp))
                }
            )
    ) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { context: Context ->
                MapLibre.getInstance(context)
                val offlineManager = OfflineMapManager.getInstance(context)

                // §58: render into a TextureView instead of MapLibre's default SurfaceView.
                // A SurfaceView draws on its own surface *behind* the app window and only shows
                // through a "hole" the view hierarchy punches for it -- inside Compose, with an
                // opaque splash composed on top of it at startup (SplashScreen.kt, 1.3 s + fade)
                // that hole wasn't re-punched once the splash went away, so the map area stayed
                // blank navy until some unrelated layout change (toggling blackout resizes the
                // status pill/FAB/drawer and adds the trail legend) forced the AndroidView to
                // lay out again. A TextureView is drawn as ordinary content inside the view
                // hierarchy, so it's visible as soon as anything is rendered, and it also
                // composes correctly under the translucent glass overlays. Small GPU cost,
                // which is the documented trade-off and irrelevant at this app's frame rates.
                val mapOptions = MapLibreMapOptions.createFromAttributes(context).textureMode(true)
                val mapLibreView = MapLibreMapView(context, mapOptions)
                mapLibreView.onCreate(null)
                mapLibreViewRef = mapLibreView

                mapLibreView.getMapAsync { map ->
                    mapLibreMapRef = map

                    // §50: gestures are enabled by MapLibre by default, but set them
                    // explicitly so free-pan/zoom/rotate is never silently dependent on that
                    // default.
                    map.uiSettings.isScrollGesturesEnabled = true
                    map.uiSettings.isZoomGesturesEnabled = true
                    map.uiSettings.isRotateGesturesEnabled = true
                    map.uiSettings.isTiltGesturesEnabled = true
                    map.uiSettings.isDoubleTapGesturesEnabled = true

                    // §57: MapLibre's own bottom-left logo and (i) attribution button sat
                    // underneath the MY LOCATION pill and the details drawer. The required
                    // credit ("© OpenMapTiles © OpenStreetMap contributors") is already shown
                    // permanently, legibly, without interaction, inside the COIMBATORE OFFLINE
                    // badge below -- which is what the OSMF attribution guidelines ask for -- so
                    // the native widgets were redundant as well as overlapping. MapLibre itself
                    // is BSD-licensed and does not require its logo to be displayed.
                    map.uiSettings.isLogoEnabled = false
                    map.uiSettings.isAttributionEnabled = false

                    // A gesture-initiated camera move (the user actually dragging/pinching the
                    // map, as opposed to our own moveCamera() calls below or the recenter
                    // buttons' own animated moves) means the user wants free view -- stop
                    // auto-recentering until they explicitly ask to recenter again.
                    map.addOnCameraMoveStartedListener { reason ->
                        if (reason == MapLibreMap.OnCameraMoveStartedListener.REASON_API_GESTURE) {
                            isFollowingUser = false
                        }
                    }

                    val initialPoint = if (latitude > 1.0 && longitude > 1.0) {
                        LatLng(latitude, longitude)
                    } else {
                        OfflineMapManager.COIMBATORE_CENTER
                    }
                    map.cameraPosition = CameraPosition.Builder()
                        .target(initialPoint)
                        .zoom(15.5)
                        .build()

                    // §60: resolve the style on a background thread -- the first call also does
                    // the one-time mbtiles/glyph copy + SQLite verify (shared with
                    // NavigationEngine, see OfflineMapManager.getInstance), which used to run
                    // synchronously right here on the main thread. setStyle() itself stays on
                    // the main thread, as MapLibre requires.
                    mapScope.launch {
                        val styleUri = withContext(Dispatchers.IO) { offlineManager.resolveStyleUri() }
                        if (styleUri == null) {
                            Log.e(TAG, "No offline style available (status=${offlineManager.getOfflineMapStatusString()}) -- vector map data has not been generated yet, see OfflineMapManager's class doc")
                        } else {
                            map.setStyle(Style.Builder().fromUri(styleUri)) { style ->
                                setUpDynamicLayers(context, style)
                                styleReady = true
                            }
                        }
                    }
                }

                mapLibreView
            },
            update = { _ ->
                if (!styleReady) return@AndroidView
                val style = mapLibreMapRef?.style ?: return@AndroidView
                val map = mapLibreMapRef ?: return@AndroidView

                val hasRealFix = latitude > 1.0 && longitude > 1.0
                val currentLatLng = if (hasRealFix) {
                    LatLng(latitude, longitude)
                } else {
                    OfflineMapManager.COIMBATORE_CENTER
                }

                if (blackoutMode && !wasBlackout.value) {
                    correctedTrail.clear()
                    naiveTrail.clear()
                }
                wasBlackout.value = blackoutMode

                if (blackoutMode) {
                    correctedTrail.add(currentLatLng)
                    if (correctedTrail.size > MAX_TRAIL_POINTS) correctedTrail.removeAt(0)
                }

                if (blackoutMode) {
                    val naivePoint = if (naiveLatitude > 1.0 && naiveLongitude > 1.0) {
                        LatLng(naiveLatitude, naiveLongitude)
                    } else {
                        currentLatLng
                    }
                    naiveTrail.add(naivePoint)
                    if (naiveTrail.size > MAX_TRAIL_POINTS) naiveTrail.removeAt(0)
                }

                // Vehicle marker -- position uses the Phase-3-smoothed lat/lon; visibility
                // (empty vs. populated GeoJSON) uses the real hasRealFix, same as before.
                val vehicleSource = style.getSourceAs<GeoJsonSource>(SRC_VEHICLE)
                if (hasRealFix) {
                    vehicleSource?.setGeoJson(pointFeatureGeoJson(displayedMarkerLat, displayedMarkerLon))
                    style.getLayerAs<SymbolLayer>(LAYER_VEHICLE)?.setProperties(
                        PropertyFactory.iconRotate(headingDeg)
                    )
                } else {
                    vehicleSource?.setGeoJson(EMPTY_FEATURE_COLLECTION)
                }

                style.getSourceAs<GeoJsonSource>(SRC_CORRECTED_TRAIL)
                    ?.setGeoJson(if (blackoutMode) lineStringGeoJson(correctedTrail) else EMPTY_FEATURE_COLLECTION)

                style.getSourceAs<GeoJsonSource>(SRC_NAIVE_TRAIL)
                    ?.setGeoJson(if (blackoutMode) lineStringGeoJson(naiveTrail) else EMPTY_FEATURE_COLLECTION)

                val uncertaintySource = style.getSourceAs<GeoJsonSource>(SRC_UNCERTAINTY)
                if (blackoutMode && uncertaintyRadiusMeters > 0.5) {
                    uncertaintySource?.setGeoJson(
                        circlePolygonGeoJson(currentLatLng.latitude, currentLatLng.longitude, animatedUncertaintyRadius.toDouble())
                    )
                } else {
                    uncertaintySource?.setGeoJson(EMPTY_FEATURE_COLLECTION)
                }

                val pinpointSource = style.getSourceAs<GeoJsonSource>(SRC_PINPOINT)
                if (hasRealFix) {
                    pinpointSource?.setGeoJson(
                        circlePolygonGeoJson(currentLatLng.latitude, currentLatLng.longitude, 8.0)
                    )
                } else {
                    pinpointSource?.setGeoJson(EMPTY_FEATURE_COLLECTION)
                }

                if (isFollowingUser) {
                    map.moveCamera(CameraUpdateFactory.newLatLng(currentLatLng))
                }
            }
        )

        // §57: all floating map controls live inside this box. Full-screen (isExpanded), it is
        // inset by the system bars (status bar / gesture bar) plus the host screen's own overlay
        // space, so nothing here draws under the clock/battery icons, the status pill, or the
        // details drawer. Plain Box with no pointer input, so map gestures still pass through.
        Box(
            modifier = Modifier
                .fillMaxSize()
                .then(if (isExpanded) Modifier.windowInsetsPadding(WindowInsets.safeDrawing) else Modifier)
                .padding(top = overlayTopPadding, bottom = overlayBottomPadding)
        ) {
            // ========================================================
            // FLOATING MAP OVERLAY CONTROLS -- unchanged visually from the pre-migration version;
            // only the camera/marker calls underneath changed from osmdroid's `controller`/
            // `overlays` API to MapLibre's `MapLibreMap`/`CameraUpdateFactory`.
            // ========================================================

            Surface(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(start = 12.dp, top = 12.dp),
                shape = RoundedCornerShape(20.dp),
                color = Color(0xEE0F172A),
                shadowElevation = 4.dp
            ) {
                Column(
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(7.dp)
                                .background(Color(0xFF10B981), CircleShape)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "COIMBATORE OFFLINE",
                            fontSize = 10.sp,
                            fontWeight = FontWeight.ExtraBold,
                            color = Color.White,
                            letterSpacing = 0.5.sp
                        )
                    }
                    // Attribution -- required, not cosmetic, and confirmed twice over: OSMF's own
                    // Attribution Guidelines require "© OpenStreetMap contributors" (legible, in a
                    // corner, visible without interaction -- satisfied by living inside an
                    // already-always-visible badge). The real vector data generated this session
                    // (PROJECT_STATUS.md's Planetiler entry) additionally printed its own required
                    // credit at build time -- "Maps made with these vector tiles must display a
                    // visible credit: © OpenMapTiles © OpenStreetMap contributors" -- and the
                    // generated file's own metadata table embeds the same two-part attribution
                    // string. Both are included below; this was missed in the original §42
                    // attribution pass since no real OpenMapTiles-schema output existed yet to
                    // reveal the additional requirement.
                    Text(
                        text = "© OpenMapTiles © OpenStreetMap contributors",
                        fontSize = 8.sp,
                        color = Color(0xFFB4BCD0),
                        modifier = Modifier.padding(top = 2.dp)
                    )
                }
            }

            // §53: trail-color legend, shown only during blackout (the only time either trail
            // draws anything). Added after a real blackout-mode test session showed the red
            // (naive/uncorrected) trail drifting far across the map while "DR Distance" and
            // "Motion" both correctly reported near-zero movement -- an internally consistent
            // result (the two lines measure genuinely different things: blue is this app's actual
            // position estimate, red is a deliberately uncorrected reference kept only to show why
            // the EKF/ZUPT/ML pipeline is needed), but with no on-screen explanation it reads as
            // the app contradicting itself. This doesn't change what either trail does -- it only
            // labels them so the red line's drift can't be mistaken for this app's own estimate.
            if (blackoutMode) {
                Surface(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(start = 12.dp, top = 74.dp),
                    shape = RoundedCornerShape(14.dp),
                    color = Color(0xEE0F172A),
                    shadowElevation = 4.dp
                ) {
                    Column(
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                        verticalArrangement = Arrangement.spacedBy(3.dp)
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .size(16.dp, 3.dp)
                                    .background(Color(0xFF2563EB), RoundedCornerShape(2.dp))
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = "Corrected (this app's estimate)",
                                fontSize = 9.sp,
                                color = Color(0xFFE2E8F0)
                            )
                        }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .size(16.dp, 3.dp)
                                    .background(Color(0xFFDC2626), RoundedCornerShape(2.dp))
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = "Uncorrected reference only",
                                fontSize = 9.sp,
                                color = Color(0xFFE2E8F0)
                            )
                        }
                    }
                }
            }

            Surface(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(12.dp),
                shape = RoundedCornerShape(20.dp),
                color = Color(0xEE0F172A),
                shadowElevation = 4.dp
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "🧭 ${headingDeg.toInt()}°",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF60A5FA)
                    )
                }
            }

            // §57: the bottom-left "📍 MY LOCATION" pill was removed -- it did exactly the same
            // thing as the 🎯 recenter button below (re-enable follow + move camera to the
            // current position at zoom 16) and occupied the corner the GNSS blackout button and
            // details drawer need. The 🎯 button keeps its blue "tap to recenter" highlight
            // whenever the user has panned away.
            if (showControls) Column(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                if (onToggleExpand != null) {
                    Surface(
                        modifier = Modifier
                            .size(38.dp)
                            .clickable { onToggleExpand() },
                        shape = CircleShape,
                        color = Color(0xFF0F172A),
                        shadowElevation = 4.dp
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            Text(if (isExpanded) "↙" else "⛶", fontSize = 16.sp, color = Color.White, fontWeight = FontWeight.Bold)
                        }
                    }
                }

                Surface(
                    modifier = Modifier
                        .size(38.dp)
                        .clickable {
                            isFollowingUser = true
                            val point = if (latitude > 1.0 && longitude > 1.0) {
                                LatLng(latitude, longitude)
                            } else {
                                OfflineMapManager.COIMBATORE_CENTER
                            }
                            mapLibreMapRef?.moveCamera(CameraUpdateFactory.newLatLngZoom(point, 16.0))
                        },
                    // §50: highlighted blue whenever the user has panned away from follow mode,
                    // as a "tap to recenter" affordance; plain white while already following.
                    shape = CircleShape,
                    color = if (isFollowingUser) Color.White else Color(0xFF2563EB),
                    shadowElevation = 4.dp
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Text("🎯", fontSize = 16.sp)
                    }
                }

                Surface(
                    modifier = Modifier
                        .size(34.dp)
                        .clickable {
                            mapLibreMapRef?.moveCamera(CameraUpdateFactory.zoomIn())
                        },
                    shape = CircleShape,
                    color = Color.White,
                    shadowElevation = 4.dp
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Text("+", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1E293B))
                    }
                }

                Surface(
                    modifier = Modifier
                        .size(34.dp)
                        .clickable {
                            mapLibreMapRef?.moveCamera(CameraUpdateFactory.zoomOut())
                        },
                    shape = CircleShape,
                    color = Color.White,
                    shadowElevation = 4.dp
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Text("-", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1E293B))
                    }
                }
            }
        }
    }
}

/**
 * One-time setup, called from the style-loaded callback: registers the vehicle marker icon
 * image and adds every source/layer this file updates on each recomposition. Nothing here
 * depends on live app state -- only `update` (in the composable above) pushes fresh data.
 */
private fun setUpDynamicLayers(context: Context, style: Style) {
    val iconDrawable = ContextCompat.getDrawable(context, R.drawable.ic_navigation_arrow)
    if (iconDrawable != null && style.getImage(ICON_VEHICLE) == null) {
        style.addImage(ICON_VEHICLE, iconDrawable.toBitmap())
    }

    style.addSource(GeoJsonSource(SRC_VEHICLE, EMPTY_FEATURE_COLLECTION))
    style.addLayer(
        SymbolLayer(LAYER_VEHICLE, SRC_VEHICLE).withProperties(
            PropertyFactory.iconImage(ICON_VEHICLE),
            PropertyFactory.iconAllowOverlap(true),
            PropertyFactory.iconIgnorePlacement(true),
            PropertyFactory.iconRotationAlignment("map")
        )
    )

    style.addSource(GeoJsonSource(SRC_CORRECTED_TRAIL, EMPTY_FEATURE_COLLECTION))
    style.addLayerBelow(
        LineLayer(LAYER_CORRECTED_TRAIL, SRC_CORRECTED_TRAIL).withProperties(
            PropertyFactory.lineColor("#2563EB"),
            PropertyFactory.lineWidth(4f),
            PropertyFactory.lineCap("round"),
            PropertyFactory.lineJoin("round")
        ),
        LAYER_VEHICLE
    )

    style.addSource(GeoJsonSource(SRC_NAIVE_TRAIL, EMPTY_FEATURE_COLLECTION))
    style.addLayerBelow(
        LineLayer(LAYER_NAIVE_TRAIL, SRC_NAIVE_TRAIL).withProperties(
            PropertyFactory.lineColor("#DC2626"),
            PropertyFactory.lineWidth(3f),
            PropertyFactory.lineOpacity(0.8f),
            PropertyFactory.lineCap("round"),
            PropertyFactory.lineJoin("round")
        ),
        LAYER_VEHICLE
    )

    style.addSource(GeoJsonSource(SRC_UNCERTAINTY, EMPTY_FEATURE_COLLECTION))
    style.addLayerBelow(
        FillLayer(LAYER_UNCERTAINTY_FILL, SRC_UNCERTAINTY).withProperties(
            PropertyFactory.fillColor("#DC2626"),
            PropertyFactory.fillOpacity(0.16f)
        ),
        LAYER_VEHICLE
    )
    style.addLayerBelow(
        LineLayer(LAYER_UNCERTAINTY_LINE, SRC_UNCERTAINTY).withProperties(
            PropertyFactory.lineColor("#DC2626"),
            PropertyFactory.lineWidth(1.5f),
            PropertyFactory.lineOpacity(0.5f)
        ),
        LAYER_VEHICLE
    )

    style.addSource(GeoJsonSource(SRC_PINPOINT, EMPTY_FEATURE_COLLECTION))
    style.addLayerBelow(
        FillLayer(LAYER_PINPOINT_FILL, SRC_PINPOINT).withProperties(
            PropertyFactory.fillColor("#2563EB"),
            PropertyFactory.fillOpacity(0.14f)
        ),
        LAYER_VEHICLE
    )
    style.addLayerBelow(
        LineLayer(LAYER_PINPOINT_LINE, SRC_PINPOINT).withProperties(
            PropertyFactory.lineColor("#2563EB"),
            PropertyFactory.lineWidth(1.5f),
            PropertyFactory.lineOpacity(0.65f)
        ),
        LAYER_VEHICLE
    )
}

private fun pointFeatureGeoJson(lat: Double, lon: Double): String =
    """{"type":"Feature","geometry":{"type":"Point","coordinates":[$lon,$lat]}}"""

private fun lineStringGeoJson(points: List<LatLng>): String {
    if (points.size < 2) return EMPTY_FEATURE_COLLECTION
    val coords = points.joinToString(",") { "[${it.longitude},${it.latitude}]" }
    return """{"type":"Feature","geometry":{"type":"LineString","coordinates":[$coords]}}"""
}

/**
 * Real geodesic circle polygon (equirectangular small-radius approximation -- entirely adequate
 * at the few-hundred-meter scale this uncertainty circle/pinpoint ring ever draws at) around a
 * center point at a given radius in meters. Ports the same math osmdroid's own
 * `Polygon.pointsAsCircle()` used, since MapLibre has no built-in equivalent -- confirmed via
 * this session's research, not assumed absent.
 */
private fun circlePolygonGeoJson(centerLat: Double, centerLon: Double, radiusMeters: Double, segments: Int = 36): String {
    if (radiusMeters <= 0.0) return EMPTY_FEATURE_COLLECTION
    val latRad = Math.toRadians(centerLat)
    val points = (0..segments).map { i ->
        val angle = 2.0 * Math.PI * i / segments
        val dLat = Math.toDegrees((radiusMeters * cos(angle)) / EARTH_RADIUS_METERS)
        val dLon = Math.toDegrees((radiusMeters * sin(angle)) / (EARTH_RADIUS_METERS * cos(latRad)))
        "[${centerLon + dLon},${centerLat + dLat}]"
    }
    return """{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[${points.joinToString(",")}]]}}"""
}
