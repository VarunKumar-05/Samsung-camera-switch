package dev.camlab

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraExtensionCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.extensions.ExtensionMode
import androidx.camera.extensions.ExtensionsManager
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat

/** Dumps everything relevant to multi-frame RAW work into a text report. */
object CapabilityProbe {

    private val CAPABILITY_NAMES = mapOf(
        0 to "BACKWARD_COMPATIBLE", 1 to "MANUAL_SENSOR", 2 to "MANUAL_POST_PROCESSING",
        3 to "RAW", 4 to "PRIVATE_REPROCESSING", 5 to "READ_SENSOR_SETTINGS",
        6 to "BURST_CAPTURE", 7 to "YUV_REPROCESSING", 8 to "DEPTH_OUTPUT",
        9 to "CONSTRAINED_HIGH_SPEED_VIDEO", 10 to "MOTION_TRACKING",
        11 to "LOGICAL_MULTI_CAMERA", 12 to "MONOCHROME", 13 to "SECURE_IMAGE_DATA",
        14 to "SYSTEM_CAMERA", 15 to "OFFLINE_PROCESSING",
        16 to "ULTRA_HIGH_RESOLUTION_SENSOR", 17 to "REMOSAIC_REPROCESSING",
        18 to "DYNAMIC_RANGE_TEN_BIT", 19 to "STREAM_USE_CASE", 20 to "COLOR_SPACE_PROFILES",
    )

    private val LEVEL_NAMES = mapOf(
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY to "LEGACY",
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED to "LIMITED",
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL to "FULL",
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 to "LEVEL_3",
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL to "EXTERNAL",
    )

    private val EXTENSION_NAMES = mapOf(
        CameraExtensionCharacteristics.EXTENSION_AUTOMATIC to "AUTOMATIC",
        CameraExtensionCharacteristics.EXTENSION_BOKEH to "BOKEH",
        CameraExtensionCharacteristics.EXTENSION_HDR to "HDR",
        CameraExtensionCharacteristics.EXTENSION_NIGHT to "NIGHT",
        @Suppress("DEPRECATION") CameraExtensionCharacteristics.EXTENSION_FACE_RETOUCH to "FACE_RETOUCH",
    )

    private val CFA_NAMES = listOf("RGGB", "GRBG", "GBRG", "BGGR", "RGB(no mosaic)")

    fun report(ctx: Context): String {
        val sb = StringBuilder()
        sb.appendLine("=== CamLab capability probe ===")
        sb.appendLine("Device: ${Build.MANUFACTURER} ${Build.MODEL} (${Build.DEVICE})")
        sb.appendLine("Android ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT}), build ${Build.DISPLAY}")
        val mgr = ctx.getSystemService(CameraManager::class.java)
        val ids = mgr.cameraIdList.toList()
        sb.appendLine("Camera ids exposed to apps: $ids")
        sb.appendLine()

        val seen = HashSet<String>()
        for (id in ids) {
            describe(mgr, id, sb, seen, indent = "")
        }
        return sb.toString()
    }

    private fun describe(mgr: CameraManager, id: String, sb: StringBuilder, seen: MutableSet<String>, indent: String) {
        if (!seen.add(id)) return
        val c: CameraCharacteristics = try {
            mgr.getCameraCharacteristics(id)
        } catch (e: Exception) {
            sb.appendLine("${indent}[camera $id] not readable: ${e.message}")
            return
        }
        val facing = when (c.get(CameraCharacteristics.LENS_FACING)) {
            CameraCharacteristics.LENS_FACING_BACK -> "BACK"
            CameraCharacteristics.LENS_FACING_FRONT -> "FRONT"
            else -> "EXTERNAL"
        }
        sb.appendLine("${indent}--- camera $id ($facing) ---")
        sb.appendLine("${indent}hardware level: ${LEVEL_NAMES[c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)]}")
        val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.toList().orEmpty()
        sb.appendLine("${indent}capabilities: ${caps.joinToString { CAPABILITY_NAMES[it] ?: "#$it" }}")
        val hasRaw = caps.contains(3)
        sb.appendLine("${indent}RAW capture supported: $hasRaw   BURST_CAPTURE: ${caps.contains(6)}   MANUAL_SENSOR: ${caps.contains(1)}")

        val pix = c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
        val active = c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
        sb.appendLine("${indent}pixel array: $pix   active array: $active   sensor orientation: ${c.get(CameraCharacteristics.SENSOR_ORIENTATION)}")
        c.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT)?.let {
            sb.appendLine("${indent}CFA: ${CFA_NAMES.getOrNull(it) ?: it}")
        }
        sb.appendLine("${indent}white level: ${c.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL)}   black level pattern: ${c.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)?.let { p -> (0..3).map { p.getOffsetForIndex(it % 2, it / 2) } }}")
        sb.appendLine("${indent}exposure range (ns): ${c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)}   ISO range: ${c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)}   max analog ISO: ${c.get(CameraCharacteristics.SENSOR_MAX_ANALOG_SENSITIVITY)}")
        sb.appendLine("${indent}focal lengths: ${c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.toList()}   apertures: ${c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)?.toList()}")
        sb.appendLine("${indent}OIS modes: ${c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)?.toList()}   min focus dist (diopters): ${c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)}")
        sb.appendLine("${indent}noise-reduction modes: ${c.get(CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES)?.toList()}   edge modes: ${c.get(CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES)?.toList()}")
        sb.appendLine("${indent}lens shading map modes: ${c.get(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_LENS_SHADING_MAP_MODES)?.toList()}")
        sb.appendLine("${indent}max pipeline depth: ${c.get(CameraCharacteristics.REQUEST_PIPELINE_MAX_DEPTH)}   max RAW streams: ${c.get(CameraCharacteristics.REQUEST_MAX_NUM_OUTPUT_RAW)}   max JPEG/YUV(proc) streams: ${c.get(CameraCharacteristics.REQUEST_MAX_NUM_OUTPUT_PROC_STALLING)}/${c.get(CameraCharacteristics.REQUEST_MAX_NUM_OUTPUT_PROC)}")

        val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        if (map != null) {
            for ((name, fmt) in listOf("RAW_SENSOR" to ImageFormat.RAW_SENSOR, "JPEG" to ImageFormat.JPEG, "YUV_420_888" to ImageFormat.YUV_420_888)) {
                val sizes = map.getOutputSizes(fmt)?.sortedByDescending { it.width.toLong() * it.height }.orEmpty()
                if (sizes.isEmpty()) {
                    sb.appendLine("${indent}$name: none")
                    continue
                }
                sb.appendLine("${indent}$name sizes: ${sizes.take(6).joinToString { "${it.width}x${it.height}" }}${if (sizes.size > 6) " …" else ""}")
                val top = sizes.first()
                sb.appendLine("${indent}  largest $name ${top.width}x${top.height}: ${rate(map.getOutputMinFrameDuration(fmt, top), map.getOutputStallDuration(fmt, top))}")
                if (fmt == ImageFormat.RAW_SENSOR && sizes.size > 1) {
                    val small = sizes.last()
                    sb.appendLine("${indent}  smallest $name ${small.width}x${small.height}: ${rate(map.getOutputMinFrameDuration(fmt, small), map.getOutputStallDuration(fmt, small))}")
                }
            }
        }
        // Samsung's 50MP sensors often expose full-resolution output only in "maximum resolution" mode.
        val mapMax = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP_MAXIMUM_RESOLUTION)
        if (mapMax != null) {
            for ((name, fmt) in listOf("RAW_SENSOR" to ImageFormat.RAW_SENSOR, "JPEG" to ImageFormat.JPEG, "YUV_420_888" to ImageFormat.YUV_420_888)) {
                val sizes: List<Size> = mapMax.getOutputSizes(fmt)?.sortedByDescending { it.width.toLong() * it.height }.orEmpty()
                if (sizes.isNotEmpty()) {
                    val top = sizes.first()
                    sb.appendLine("${indent}MAX-RES mode $name: ${sizes.take(3).joinToString { "${it.width}x${it.height}" }}  ${rate(mapMax.getOutputMinFrameDuration(fmt, top), mapMax.getOutputStallDuration(fmt, top))}")
                }
            }
        }

        // Vendor extensions (Camera2 framework API; CameraX Extensions wraps the same thing).
        try {
            val ext = mgr.getCameraExtensionCharacteristics(id).supportedExtensions
            sb.appendLine("${indent}Camera2 extensions: ${if (ext.isEmpty()) "none" else ext.joinToString { EXTENSION_NAMES[it] ?: "#$it" }}")
        } catch (e: Exception) {
            sb.appendLine("${indent}Camera2 extensions: query failed (${e.javaClass.simpleName})")
        }

        val physical = try { c.physicalCameraIds } catch (e: Exception) { emptySet() }
        if (physical.isNotEmpty()) {
            sb.appendLine("${indent}logical camera backed by physical ids: $physical")
            for (pid in physical) describe(mgr, pid, sb, seen, "$indent    ")
        }
        sb.appendLine()
    }

    private fun rate(minFrameNs: Long, stallNs: Long): String {
        if (minFrameNs <= 0) return "frame duration unknown"
        val fps = 1e9 / minFrameNs
        val sustained = 1e9 / (minFrameNs + stallNs.coerceAtLeast(0))
        return "min frame %.1f ms (max %.1f fps), stall %.1f ms (≈%.1f fps sustained)".format(minFrameNs / 1e6, fps, stallNs / 1e6, sustained)
    }

    /** CameraX Extensions availability for the default back camera. Async; reports through [done]. */
    fun probeCameraXExtensions(ctx: Context, done: (String) -> Unit) {
        try {
            val providerFuture = ProcessCameraProvider.getInstance(ctx)
            providerFuture.addListener({
                try {
                    val provider = providerFuture.get()
                    val extFuture = ExtensionsManager.getInstanceAsync(ctx, provider)
                    extFuture.addListener({
                        try {
                            val em = extFuture.get()
                            val sel = CameraSelector.DEFAULT_BACK_CAMERA
                            val modes = listOf(
                                "AUTO" to ExtensionMode.AUTO,
                                "HDR" to ExtensionMode.HDR,
                                "NIGHT" to ExtensionMode.NIGHT,
                                "BOKEH" to ExtensionMode.BOKEH,
                                "FACE_RETOUCH" to ExtensionMode.FACE_RETOUCH,
                            )
                            val line = modes.joinToString("   ") { (n, m) -> "$n=${em.isExtensionAvailable(sel, m)}" }
                            done("CameraX Extensions (default back camera): $line")
                        } catch (e: Exception) {
                            done("CameraX Extensions: failed (${e.javaClass.simpleName}: ${e.message})")
                        }
                    }, ContextCompat.getMainExecutor(ctx))
                } catch (e: Exception) {
                    done("CameraX provider: failed (${e.javaClass.simpleName}: ${e.message})")
                }
            }, ContextCompat.getMainExecutor(ctx))
        } catch (e: CameraAccessException) {
            done("CameraX Extensions: camera access error ${e.message}")
        } catch (e: Exception) {
            done("CameraX Extensions: ${e.javaClass.simpleName}: ${e.message}")
        }
    }
}
