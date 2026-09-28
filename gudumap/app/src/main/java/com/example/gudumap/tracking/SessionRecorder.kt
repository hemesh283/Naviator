package com.example.gudumap.tracking

import com.example.gudumap.navigation.NavigationState
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * §60: one recorded moment of a navigation session -- what was on screen, not raw sensor data.
 * `naive*` and `gps*` are only present during a blackout (that's the only time the uncorrected
 * trail exists, and the only time GPS is an independent reference rather than an input).
 */
data class SessionFrame(
    val wallTimeMs: Long,
    val blackout: Boolean,
    val latitude: Double,
    val longitude: Double,
    val naiveLatitude: Double?,
    val naiveLongitude: Double?,
    val gpsLatitude: Double?,
    val gpsLongitude: Double?,
    val uncertaintyMeters: Double,
    val headingDeg: Float,
    val speedKmh: Float,
    val drDistanceMeters: Double
)

/**
 * §60: records [NavigationState] snapshots at a fixed max rate while recording is on. Pure
 * observer -- reads the state the UI already receives, never touches the engine or the model.
 */
class SessionRecorder(
    private val minIntervalMs: Long = 500L,  // 2 frames/s is plenty for trails at walk/drive speeds
    private val maxFrames: Int = 30_000      // ~4 h at 2 Hz; recording just stops growing after that
) {
    private val frames = ArrayList<SessionFrame>()
    private var lastFrameMs = 0L

    var isRecording: Boolean = false
        private set
    var startedAtMs: Long = 0L
        private set

    val frameCount: Int get() = frames.size

    fun start(nowMs: Long = System.currentTimeMillis()) {
        frames.clear()
        lastFrameMs = 0L
        startedAtMs = nowMs
        isRecording = true
    }

    /** Returns true if a frame was added. */
    fun onState(state: NavigationState, nowMs: Long = System.currentTimeMillis()): Boolean {
        if (!isRecording) return false
        if (state.latitude == 0.0 && state.longitude == 0.0) return false // no fix yet
        if (lastFrameMs != 0L && nowMs - lastFrameMs < minIntervalMs) return false
        if (frames.size >= maxFrames) return false
        lastFrameMs = nowMs

        val hasNaive = state.blackoutMode && (state.naiveLatitude != 0.0 || state.naiveLongitude != 0.0)
        frames.add(
            SessionFrame(
                wallTimeMs = nowMs,
                blackout = state.blackoutMode,
                latitude = state.latitude,
                longitude = state.longitude,
                naiveLatitude = if (hasNaive) state.naiveLatitude else null,
                naiveLongitude = if (hasNaive) state.naiveLongitude else null,
                gpsLatitude = if (state.blackoutMode) state.gnssGroundTruthLat else null,
                gpsLongitude = if (state.blackoutMode) state.gnssGroundTruthLon else null,
                uncertaintyMeters = state.uncertaintyRadiusMeters,
                headingDeg = state.headingDeg,
                speedKmh = state.speedKmh,
                drDistanceMeters = if (state.blackoutMode) state.blackoutMetrics.drDistance else 0.0
            )
        )
        return true
    }

    fun stop(): List<SessionFrame> {
        isRecording = false
        return frames.toList()
    }
}

/**
 * §60: session files live in `filesDir/sessions/` as CSV (the replayable source of truth) with a
 * GPX generated next to it on export. Shared with other apps only through the FileProvider
 * declared in AndroidManifest.xml (`res/xml/file_paths.xml`).
 */
object SessionFiles {
    private const val DIR_NAME = "sessions"
    private const val CSV_HEADER =
        "wall_time_ms,blackout,lat,lon,naive_lat,naive_lon,gps_lat,gps_lon,uncertainty_m,heading_deg,speed_kmh,dr_distance_m"

    fun dir(filesDir: File): File = File(filesDir, DIR_NAME).apply { if (!exists()) mkdirs() }

    fun newCsvFile(filesDir: File, startedAtMs: Long): File {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date(startedAtMs))
        return File(dir(filesDir), "gudumap_session_$stamp.csv")
    }

    fun latestCsv(filesDir: File): File? =
        dir(filesDir).listFiles { f -> f.isFile && f.name.endsWith(".csv") }?.maxByOrNull { it.lastModified() }

    fun gpxFor(csv: File): File = File(csv.parentFile, csv.nameWithoutExtension + ".gpx")

    private fun num(v: Double?): String = if (v == null) "" else String.format(Locale.US, "%.7f", v)

    fun writeCsv(file: File, frames: List<SessionFrame>) {
        file.bufferedWriter(Charsets.UTF_8).use { w ->
            w.write(CSV_HEADER)
            w.newLine()
            for (f in frames) {
                w.write(
                    listOf(
                        f.wallTimeMs.toString(),
                        if (f.blackout) "1" else "0",
                        num(f.latitude), num(f.longitude),
                        num(f.naiveLatitude), num(f.naiveLongitude),
                        num(f.gpsLatitude), num(f.gpsLongitude),
                        String.format(Locale.US, "%.2f", f.uncertaintyMeters),
                        String.format(Locale.US, "%.1f", f.headingDeg),
                        String.format(Locale.US, "%.2f", f.speedKmh),
                        String.format(Locale.US, "%.2f", f.drDistanceMeters)
                    ).joinToString(",")
                )
                w.newLine()
            }
        }
    }

    /** Parses a file written by [writeCsv]; malformed lines are skipped rather than failing. */
    fun readCsv(file: File): List<SessionFrame> {
        val out = ArrayList<SessionFrame>()
        file.bufferedReader(Charsets.UTF_8).useLines { lines ->
            for (line in lines) {
                if (line.isBlank() || line.startsWith("wall_time_ms")) continue
                val c = line.split(",")
                if (c.size < 12) continue
                val time = c[0].toLongOrNull() ?: continue
                val lat = c[2].toDoubleOrNull() ?: continue
                val lon = c[3].toDoubleOrNull() ?: continue
                out.add(
                    SessionFrame(
                        wallTimeMs = time,
                        blackout = c[1] == "1",
                        latitude = lat,
                        longitude = lon,
                        naiveLatitude = c[4].toDoubleOrNull(),
                        naiveLongitude = c[5].toDoubleOrNull(),
                        gpsLatitude = c[6].toDoubleOrNull(),
                        gpsLongitude = c[7].toDoubleOrNull(),
                        uncertaintyMeters = c[8].toDoubleOrNull() ?: 0.0,
                        headingDeg = c[9].toFloatOrNull() ?: 0f,
                        speedKmh = c[10].toFloatOrNull() ?: 0f,
                        drDistanceMeters = c[11].toDoubleOrNull() ?: 0.0
                    )
                )
            }
        }
        return out
    }

    /**
     * GPX 1.1 with up to three tracks: the app's own estimate (whole session), the independent
     * GPS reference (blackout periods only), and the uncorrected naive trail (blackout only).
     * Opens in Google Earth, gpx.studio, QGIS, etc.
     */
    fun writeGpx(file: File, frames: List<SessionFrame>) {
        val iso = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }
        fun pt(lat: Double, lon: Double, t: Long): String =
            String.format(Locale.US, "      <trkpt lat=\"%.7f\" lon=\"%.7f\"><time>%s</time></trkpt>\n", lat, lon, iso.format(Date(t)))

        // Split a track into segments wherever its data is absent (e.g. between two blackouts).
        fun segments(pick: (SessionFrame) -> Pair<Double, Double>?): List<List<String>> {
            val segs = ArrayList<List<String>>()
            var cur = ArrayList<String>()
            for (f in frames) {
                val p = pick(f)
                if (p == null) {
                    if (cur.isNotEmpty()) { segs.add(cur); cur = ArrayList() }
                } else {
                    cur.add(pt(p.first, p.second, f.wallTimeMs))
                }
            }
            if (cur.isNotEmpty()) segs.add(cur)
            return segs
        }

        val tracks = listOf(
            "Gudumap estimate (EKF + dead reckoning)" to segments { Pair(it.latitude, it.longitude) },
            "GPS reference (during blackout)" to segments { f ->
                val la = f.gpsLatitude; val lo = f.gpsLongitude
                if (f.blackout && la != null && lo != null) Pair(la, lo) else null
            },
            "Uncorrected reference (naive, during blackout)" to segments { f ->
                val la = f.naiveLatitude; val lo = f.naiveLongitude
                if (f.blackout && la != null && lo != null) Pair(la, lo) else null
            }
        )

        file.bufferedWriter(Charsets.UTF_8).use { w ->
            w.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            w.write("<gpx version=\"1.1\" creator=\"Gudumap (SIH26168)\" xmlns=\"http://www.topografix.com/GPX/1/1\">\n")
            for ((name, segs) in tracks) {
                if (segs.isEmpty()) continue
                w.write("  <trk>\n    <name>$name</name>\n")
                for (seg in segs) {
                    w.write("    <trkseg>\n")
                    for (line in seg) w.write(line)
                    w.write("    </trkseg>\n")
                }
                w.write("  </trk>\n")
            }
            w.write("</gpx>\n")
        }
    }
}
