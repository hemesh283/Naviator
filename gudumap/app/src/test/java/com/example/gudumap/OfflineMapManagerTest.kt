package com.example.gudumap

import com.example.gudumap.map.OfflineMapManager
import com.example.gudumap.map.OfflineMapStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class OfflineMapManagerTest {

    @Test
    fun testCoimbatoreCoordinatesAndConstants() {
        assertEquals(11.0168, OfflineMapManager.COIMBATORE_DEFAULT_LAT, 1e-4)
        assertEquals(76.9558, OfflineMapManager.COIMBATORE_DEFAULT_LON, 1e-4)
        assertEquals(11, OfflineMapManager.MIN_ZOOM)
        assertEquals(16, OfflineMapManager.MAX_ZOOM)

        assertNotNull(OfflineMapManager.COIMBATORE_BOUNDS)
        assertTrue(OfflineMapManager.COIMBATORE_BOUNDS.north > OfflineMapManager.COIMBATORE_BOUNDS.south)
        assertTrue(OfflineMapManager.COIMBATORE_BOUNDS.east > OfflineMapManager.COIMBATORE_BOUNDS.west)

        // Verify bounding box covers central Coimbatore
        assertTrue(OfflineMapManager.COIMBATORE_DEFAULT_LAT < OfflineMapManager.COIMBATORE_BOUNDS.north)
        assertTrue(OfflineMapManager.COIMBATORE_DEFAULT_LAT > OfflineMapManager.COIMBATORE_BOUNDS.south)
        assertTrue(OfflineMapManager.COIMBATORE_DEFAULT_LON < OfflineMapManager.COIMBATORE_BOUNDS.east)
        assertTrue(OfflineMapManager.COIMBATORE_DEFAULT_LON > OfflineMapManager.COIMBATORE_BOUNDS.west)
    }

    @Test
    fun testOfflineMapStatusEnum() {
        val statuses = OfflineMapStatus.values()
        assertTrue(statuses.contains(OfflineMapStatus.AVAILABLE))
        assertTrue(statuses.contains(OfflineMapStatus.LOADING))
        assertTrue(statuses.contains(OfflineMapStatus.ERROR))
        assertTrue(statuses.contains(OfflineMapStatus.NOT_AVAILABLE))
    }
}
