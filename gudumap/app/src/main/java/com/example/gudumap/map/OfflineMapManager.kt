package com.example.gudumap.map

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.util.Log
import org.maplibre.android.geometry.LatLng
import java.io.File
import java.io.FileOutputStream

/**
 * Plain, library-independent bounding box for the Coimbatore demo region -- kept as a simple
 * data holder rather than a MapLibre `LatLngBounds` since nothing currently reads it (confirmed
 * via grep before this migration; it was equally unused in the pre-migration osmdroid version),
 * so there's no reason to add unverified MapLibre API surface for a value nothing consumes.
 */
data class GeoBounds(val north: Double, val east: Double, val south: Double, val west: Double)

enum class OfflineMapStatus {
    AVAILABLE,
    LOADING,
    ERROR,
    NOT_AVAILABLE
}

/**
 * Manages the offline map DATA lifecycle -- locating, copying, and verifying the bundled
 * Coimbatore map package, and resolving a MapLibre-ready style URI from it. Rendering itself
 * (MapLibre Native, replacing osmdroid) lives in `ui/components/MapView.kt`; this class only
 * ever hands that file a local path/URI, never touches rendering APIs directly -- same
 * separation of concerns the pre-migration version had (it built an osmdroid-specific
 * `MapTileProviderArray`; this version builds the MapLibre-equivalent `mbtiles://<abs-path>`
 * source URI and a resolved local style.json instead).
 *
 * §42 migration note: `coimbatore.mbtiles` is now expected to be a VECTOR-tile MBTiles file
 * (MBTiles' `tiles` table schema is identical for raster and vector; only the metadata
 * `format` row differs -- `pbf` instead of `png` -- and MapLibre reads that itself, this class
 * doesn't need to branch on it). `mbtiles://<abs-path>` (single scheme prefix directly followed
 * by the absolute filesystem path, no doubled `file://`) is MapLibre Native's real local
 * vector-tile source URI scheme -- verified by reading `MBTilesFileSource::request()` in the
 * exact `org.maplibre.gl:android-sdk:13.6.1` source (tag `android-v13.6.1`,
 * platform/default/src/mln/storage/mbtiles_file_source.cpp): it strips only the literal
 * `"mbtiles://"` prefix and requires what remains to satisfy `is_absolute_path()` (starts with
 * `/`). A `mbtiles://file://<path>` URL fails that check -- `"file://<path>"` doesn't start
 * with `/` -- and the request is rejected up front with "MBTilesFileSource only supports
 * absolute path urls", before the style's tile source ever loads. That was this method's actual
 * bug and the reason the map rendered blank (background layer only, no vector data) on a real
 * device. Every method below is written to be
 * correct once a real file is dropped in at the same asset path the old raster file used; until
 * then, `status` will read `NOT_AVAILABLE` on a real device, exactly like the pre-fix state
 * `initializeOfflineMap()` already handles honestly.
 */
class OfflineMapManager(private val context: Context) {

    companion object {
        private const val TAG = "Gudumap:OfflineMap"
        const val COIMBATORE_DEFAULT_LAT = 11.0168
        const val COIMBATORE_DEFAULT_LON = 76.9558

        // §42/§44: vector tiles self-overzoom cleanly past their native max zoom (MapLibre keeps
        // reusing the highest-zoom tile data and scales it up), unlike the old raster file
        // where zooming past MAX_ZOOM showed nothing. These bounds are therefore the app's own
        // chosen INTERACTIVE zoom range, not a hard ceiling forced by the data the way the old
        // raster MIN_ZOOM/MAX_ZOOM comment described. Re-verified directly against the real
        // generated file (`SELECT value FROM metadata WHERE name='maxzoom'`): the real
        // Planetiler/OpenMapTiles output's native data goes from minzoom=0 to maxzoom=14 --
        // confirming the guess this comment made before real data existed. MAX_ZOOM=16 below is
        // therefore two levels of overzoom past the real native data (still correct, real
        // vector-tile behavior, just less crisp than z14's own native detail at 15-16), kept for
        // continuity with the existing demo's interactive zoom range rather than dropped to 14.
        const val MIN_ZOOM = 11
        const val MAX_ZOOM = 16

        // Bounding box for Coimbatore metropolitan area (23.3 km x 20.7 km, ~482 sq km) --
        // unchanged from the raster version; this is the area any newly-generated vector
        // extract must cover at minimum.
        val COIMBATORE_BOUNDS = GeoBounds(north = 11.125, east = 77.070, south = 10.915, west = 76.880)
        val COIMBATORE_CENTER = LatLng(COIMBATORE_DEFAULT_LAT, COIMBATORE_DEFAULT_LON)

        @Volatile
        private var sharedInstance: OfflineMapManager? = null

        /**
         * §60: one process-wide instance. Previously both NavigationEngine and MapView built
         * their own, and each one's constructor copied/verified the ~MB mbtiles file and the
         * glyph files synchronously -- twice per launch, on the main thread, and on a fresh
         * install the two copies could overlap on the same target file. Both callers now share
         * this instance; the actual work happens once, in [ensureInitialized], off the main
         * thread.
         */
        fun getInstance(context: Context): OfflineMapManager =
            sharedInstance ?: synchronized(this) {
                sharedInstance ?: OfflineMapManager(context.applicationContext).also { sharedInstance = it }
            }
    }

    @Volatile
    var status: OfflineMapStatus = OfflineMapStatus.LOADING
        private set

    @Volatile
    private var initialized = false

    private var localMapFile: File? = null
    private var tileCount: Int = 0
    private var fontsDir: File? = null

    /**
     * §60: runs the copy/verify work exactly once. No longer called from `init` -- callers run it
     * on a background thread (NavigationEngine.start() and MapView's style loader both do).
     * `@Synchronized` on the same instance lock as [initializeOfflineMap], so a second caller
     * arriving mid-copy simply waits for the first to finish instead of copying again.
     */
    @Synchronized
    fun ensureInitialized() {
        if (initialized) return
        initializeOfflineMap()
        initialized = true
    }

    /**
     * Initializes the offline map dataset -- unchanged logic from the raster version:
     * 1. Locates or creates the internal storage directory: context.filesDir/maps/coimbatore/
     * 2. Copies coimbatore.mbtiles from APK assets on first run or when updated
     * 3. Verifies SQLite integrity and tile counts
     * 4. Updates status to AVAILABLE, ERROR, or NOT_AVAILABLE
     *
     * Copying to real device storage first (rather than reading straight out of `assets/`) is
     * required either way, for two independent reasons that both point to the same fix:
     * `SQLiteDatabase.openDatabase()` needs a real filesystem path (assets live inside the APK's
     * zip container, not as standalone files), and -- confirmed this session -- MapLibre's own
     * PMTiles docs note the same class of limitation for byte-range reads from `asset://`
     * sources, which is part of why this session's research concluded a copied-out MBTiles file
     * over `mbtiles://<abs-path>` is the better-supported path than PMTiles for this app anyway.
     */
    @Synchronized
    fun initializeOfflineMap() {
        status = OfflineMapStatus.LOADING
        try {
            val mapsDir = File(context.filesDir, "maps/coimbatore")
            if (!mapsDir.exists()) mapsDir.mkdirs()

            val targetFile = File(mapsDir, "coimbatore.mbtiles")

            val assetPath = try {
                context.assets.open("maps/coimbatore/coimbatore.mbtiles").close()
                "maps/coimbatore/coimbatore.mbtiles"
            } catch (e: Exception) {
                try {
                    context.assets.open("maps/coimbatore.mbtiles").close()
                    "maps/coimbatore.mbtiles"
                } catch (e2: Exception) {
                    null
                }
            }

            if (assetPath == null && !targetFile.exists()) {
                Log.e(TAG, "Offline map asset not found in assets/maps/ (expected a vector-tile coimbatore.mbtiles -- see class doc: not yet generated, needs Java 21+ and Planetiler on a real machine)")
                status = OfflineMapStatus.NOT_AVAILABLE
                return
            }

            // §26: re-copy if the bundled asset's size differs from what's already sitting in
            // internal storage, not just when the target is missing/empty -- unchanged.
            val assetSize = if (assetPath != null) {
                try {
                    context.assets.openFd(assetPath).length
                } catch (e: Exception) {
                    try {
                        context.assets.open(assetPath).use { it.available().toLong() }
                    } catch (e2: Exception) {
                        0L
                    }
                }
            } else 0L

            val shouldCopy = assetPath != null && (
                !targetFile.exists() ||
                targetFile.length() == 0L ||
                (assetSize > 0L && targetFile.length() != assetSize)
            )

            if (shouldCopy && assetPath != null) {
                Log.i(TAG, "Copying offline Coimbatore map package from $assetPath to ${targetFile.absolutePath} (asset size: ${assetSize} B)...")
                context.assets.open(assetPath).use { input ->
                    FileOutputStream(targetFile).use { output ->
                        input.copyTo(output)
                    }
                }
                Log.i(TAG, "Extracted offline map successfully (${targetFile.length() / (1024 * 1024)} MB).")
            }

            localMapFile = targetFile
            copyGlyphs()

            if (targetFile.exists() && targetFile.length() > 0L) {
                val isValid = verifyDatabase(targetFile)
                if (isValid) {
                    status = OfflineMapStatus.AVAILABLE
                    Log.i(TAG, "Offline Coimbatore map is AVAILABLE. Tile count = $tileCount, size = ${targetFile.length() / (1024 * 1024)} MB.")
                } else {
                    status = OfflineMapStatus.ERROR
                    Log.e(TAG, "Offline map file exists but verification failed.")
                }
            } else {
                status = OfflineMapStatus.NOT_AVAILABLE
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error initializing offline map: ${e.message}", e)
            status = OfflineMapStatus.ERROR
        }
    }

    /**
     * Copies the bundled offline SDF glyph ranges (assets/fonts/<fontstack>/<range>.pbf --
     * generated with Mapbox's `fontnik` from Noto Sans Regular, covering Unicode ranges 0-255
     * and 256-511: Basic Latin, Latin-1 Supplement, and Latin Extended-A/B, sufficient for the
     * English-tagged OSM names this demo region actually has) into internal storage, mirroring
     * `initializeOfflineMap()`'s own copy-then-reference pattern for `coimbatore.mbtiles` --
     * for the identical reason: MapLibre's local `file://` resource loader needs a real
     * filesystem path, not a path inside the APK's asset zip. Failure here is logged but
     * non-fatal: `resolveStyleUri()` still substitutes a (possibly-then-broken) glyphs URL
     * either way, so the map's tiles/roads still render even if labels silently don't --
     * fail-soft, matching this class's existing `status` handling for the mbtiles file itself.
     */
    private fun copyGlyphs() {
        try {
            val fontStack = "Noto Sans Regular"
            val assetDir = "fonts/$fontStack"
            val entries = try {
                context.assets.list(assetDir)
            } catch (e: Exception) {
                null
            }
            if (entries.isNullOrEmpty()) {
                Log.w(TAG, "copyGlyphs: no glyph assets found at assets/$assetDir -- road/place labels will not render")
                return
            }

            val targetDir = File(File(context.filesDir, "maps/coimbatore/fonts"), fontStack)
            if (!targetDir.exists()) targetDir.mkdirs()

            var copied = 0
            for (name in entries) {
                if (!name.endsWith(".pbf")) continue
                val assetPath = "$assetDir/$name"
                val targetFile = File(targetDir, name)
                val assetSize = try {
                    context.assets.openFd(assetPath).length
                } catch (e: Exception) {
                    try {
                        context.assets.open(assetPath).use { it.available().toLong() }
                    } catch (e2: Exception) {
                        0L
                    }
                }
                val shouldCopy = !targetFile.exists() || targetFile.length() == 0L ||
                    (assetSize > 0L && targetFile.length() != assetSize)
                if (shouldCopy) {
                    context.assets.open(assetPath).use { input ->
                        FileOutputStream(targetFile).use { output -> input.copyTo(output) }
                    }
                }
                copied++
            }

            fontsDir = File(context.filesDir, "maps/coimbatore/fonts")
            Log.i(TAG, "copyGlyphs: $copied glyph range file(s) available for fontstack \"$fontStack\" at ${fontsDir?.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "copyGlyphs: failed to copy glyph assets: ${e.message}", e)
        }
    }

    /**
     * SQLite-level integrity check only -- the `tiles` table exists in every valid MBTiles
     * file regardless of whether it holds raster or vector (pbf) tile blobs, so this check is
     * unchanged from the raster version and needed no update for the vector migration.
     */
    private fun verifyDatabase(file: File): Boolean {
        var db: SQLiteDatabase? = null
        return try {
            db = SQLiteDatabase.openDatabase(file.absolutePath, null, SQLiteDatabase.OPEN_READONLY)
            val cursor = db.rawQuery("SELECT COUNT(*) FROM tiles", null)
            var count = 0
            if (cursor.moveToFirst()) {
                count = cursor.getInt(0)
            }
            cursor.close()
            tileCount = count
            count > 0
        } catch (e: Exception) {
            Log.e(TAG, "SQLite validation error: ${e.message}", e)
            false
        } finally {
            try { db?.close() } catch (_: Exception) {}
        }
    }

    fun getLocalMapFile(): File? = localMapFile

    fun getTileCount(): Int = tileCount

    fun isOfflineMapAvailable(): Boolean = status == OfflineMapStatus.AVAILABLE

    /**
     * The MapLibre vector-tile source URI for the bundled MBTiles file, or null if it isn't
     * available. Replaces the old `createOfflineTileProvider()` (which built an osmdroid-
     * specific `MapTileProviderArray`) -- MapLibre doesn't need a provider object, just this
     * URI string referenced from a style's source `url`. The scheme is `mbtiles://` directly
     * followed by the absolute filesystem path -- NOT `mbtiles://file://<path>`, which is what
     * this method returned before and is rejected by MapLibre's `MBTilesFileSource` (see the
     * class doc above); `file.absolutePath` already starts with `/`, so concatenating it after
     * `mbtiles://` alone yields the correct `mbtiles:///data/user/0/...` form.
     */
    fun getMbtilesSourceUri(): String? {
        val file = localMapFile ?: return null
        if (!file.exists()) return null
        return "mbtiles://${file.absolutePath}"
    }

    /**
     * The `glyphs` template URL for the style JSON's root-level `glyphs` field: `file://` +
     * the internal-storage fonts directory + the standard `{fontstack}/{range}.pbf` template
     * MapLibre substitutes itself (`Resource::glyphs()` in the core, confirmed by reading it
     * alongside the `mbtiles://` fix above -- it percent-encodes `{fontstack}` and fills in
     * `{range}` before dispatch, and `LocalFileSource` percent-decodes the same way `file://`
     * paths already do for the style and mbtiles URIs, so no extra encoding is needed here).
     * Returns a template even if glyph copying actually failed (see `copyGlyphs()`) -- a
     * missing font file just means unresolved glyph requests get logged and no labels render,
     * not a broken style.json, so the map's tiles/roads keep working either way.
     */
    fun getGlyphsUrlTemplate(): String {
        val dir = fontsDir ?: File(context.filesDir, "maps/coimbatore/fonts")
        return "file://${dir.absolutePath}/{fontstack}/{range}.pbf"
    }

    /**
     * Resolves the app's style template (bundled at assets/maps/coimbatore/style_template.json,
     * authored in this app's own Color.kt navy/cyan/violet palette -- see MapView.kt's usage
     * site and this session's PROJECT_STATUS.md entry for why a literal light Organic-Maps-style
     * palette was deliberately not used) into a real style.json with the actual runtime mbtiles
     * path substituted in, writes it to internal storage, and returns a `file://` URI to it.
     *
     * Done this way rather than `Style.Builder().fromJson(String)` specifically because this
     * session could not verify that method's exact availability/signature against MapLibre's
     * real KDoc (no compiler here to catch a mistake) -- `fromUri("file://...")` uses the exact
     * same URI scheme already confirmed for the tile source itself, so it carries no equivalent
     * uncertainty.
     */
    fun resolveStyleUri(): String? {
        ensureInitialized()
        val mbtilesUri = getMbtilesSourceUri() ?: return null
        return try {
            val templateJson = try {
                context.assets.open("maps/coimbatore/style_template.json")
                    .bufferedReader(Charsets.UTF_8).use { it.readText() }
            } catch (e: Exception) {
                Log.e(TAG, "resolveStyleUri: style_template.json not found in assets/maps/coimbatore/: ${e.message}")
                return null
            }

            val resolvedJson = templateJson
                .replace("{{MBTILES_URL}}", mbtilesUri)
                .replace("{{GLYPHS_URL}}", getGlyphsUrlTemplate())

            val mapsDir = File(context.filesDir, "maps/coimbatore")
            if (!mapsDir.exists()) mapsDir.mkdirs()
            val styleFile = File(mapsDir, "style.json")
            styleFile.writeText(resolvedJson, Charsets.UTF_8)

            "file://${styleFile.absolutePath}"
        } catch (e: Exception) {
            Log.e(TAG, "resolveStyleUri: failed to resolve style: ${e.message}", e)
            null
        }
    }

    fun isOnline(): Boolean {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager ?: return false
        val network = cm.activeNetwork ?: return false
        val capabilities = cm.getNetworkCapabilities(network) ?: return false
        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
                capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }

    fun getMapStatus(blackoutMode: Boolean = true): String {
        return "OFFLINE"
    }

    fun getOfflineMapStatusString(): String {
        return when (status) {
            OfflineMapStatus.AVAILABLE -> "AVAILABLE"
            OfflineMapStatus.LOADING -> "LOADING"
            OfflineMapStatus.ERROR -> "ERROR"
            OfflineMapStatus.NOT_AVAILABLE -> "NOT_AVAILABLE"
        }
    }
}
