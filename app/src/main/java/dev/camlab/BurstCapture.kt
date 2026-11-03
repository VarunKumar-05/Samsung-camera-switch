package dev.camlab

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureFailure
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.DngCreator
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.Image
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Rational
import android.view.Surface
import android.view.TextureView
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executor
import java.util.concurrent.Executors

/**
 * Camera2 burst engine. Previews, then captures N frames with *fixed* exposure/ISO/white balance
 * (so frames are mergeable) and processing (noise reduction, edge enhancement) turned off where possible.
 * RAW mode writes DNGs + meta.json; JPEG mode writes JPEGs + meta.json.
 */
class BurstCapture(
    private val ctx: Context,
    private val previewView: TextureView,
    private val log: (String) -> Unit,
) {
    enum class Mode { RAW, JPEG }

    private val mgr = ctx.getSystemService(CameraManager::class.java)
    private val thread = HandlerThread("camlab-camera").apply { start() }
    private val handler = Handler(thread.looper)
    private val handlerExecutor = Executor { handler.post(it) }
    private val io = Executors.newSingleThreadExecutor()

    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var previewSurface: Surface? = null
    private var chars: CameraCharacteristics? = null
    private var cameraId = ""
    @Volatile private var lastPreview: TotalCaptureResult? = null

    var mode = Mode.RAW
        private set
    var burstCount = 6

    // Burst bookkeeping (touched only on the camera handler thread).
    private val results = ArrayList<TotalCaptureResult>()
    private val images = ArrayList<Image>()
    private var failed = 0
    private var lost = 0
    private var expected = 0
    private var capturing = false
    private var burstExposureNs = 0L
    private var burstIso = 0

    /** Opens the best back camera that can output [newMode], and starts the preview. */
    fun open(newMode: Mode) {
        close()
        mode = newMode
        val fmt = formatFor(newMode)
        val pick = mgr.cameraIdList
            .mapNotNull { id ->
                val c = mgr.getCameraCharacteristics(id)
                if (c.get(CameraCharacteristics.LENS_FACING) != CameraCharacteristics.LENS_FACING_BACK) return@mapNotNull null
                val sizes = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)?.getOutputSizes(fmt)
                if (sizes.isNullOrEmpty()) return@mapNotNull null
                val pa = c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
                val area = if (pa != null) pa.width.toLong() * pa.height else 0L
                Triple(id, c, area)
            }
            .maxByOrNull { it.third }
        if (pick == null) {
            log("No back camera exposes $newMode output to apps. Run the probe and send me the report.")
            return
        }
        cameraId = pick.first
        chars = pick.second
        log("Using camera $cameraId for $newMode")
        openDevice()
    }

    @SuppressLint("MissingPermission")
    private fun openDevice() {
        mgr.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                device = camera
                createSession(camera)
            }
            override fun onDisconnected(camera: CameraDevice) { camera.close(); device = null }
            override fun onError(camera: CameraDevice, error: Int) {
                log("Camera error $error"); camera.close(); device = null
            }
        }, handler)
    }

    private fun formatFor(m: Mode) = if (m == Mode.RAW) ImageFormat.RAW_SENSOR else ImageFormat.JPEG

    private fun createSession(dev: CameraDevice) {
        val c = chars ?: return
        val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return
        val fmt = formatFor(mode)
        val capSize = map.getOutputSizes(fmt).maxByOrNull { it.width.toLong() * it.height } ?: return

        // Preview: largest size (<= ~2MP) matching the capture aspect ratio.
        val ar = capSize.width.toDouble() / capSize.height
        val candidates = map.getOutputSizes(android.graphics.SurfaceTexture::class.java)
            .filter { it.width.toLong() * it.height <= 1920L * 1440 }
        val preview = candidates.filter { Math.abs(it.width.toDouble() / it.height - ar) < 0.02 }
            .maxByOrNull { it.width.toLong() * it.height }
            ?: candidates.maxByOrNull { it.width.toLong() * it.height } ?: return

        val rd = ImageReader.newInstance(capSize.width, capSize.height, fmt, (burstCount + 2).coerceAtMost(12))
        rd.setOnImageAvailableListener({ r ->
            val img = r.acquireNextImage() ?: return@setOnImageAvailableListener
            if (capturing) { images.add(img); maybeFinalize() } else img.close()
        }, handler)
        reader = rd

        val st = previewView.surfaceTexture ?: run { log("Preview surface not ready"); return }
        st.setDefaultBufferSize(preview.width, preview.height)
        val ps = Surface(st)
        previewSurface = ps

        val orient = c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        previewView.post {
            val lp = previewView.layoutParams
            lp.height = if (orient % 180 == 90) previewView.width * preview.width / preview.height
                        else previewView.width * preview.height / preview.width
            previewView.layoutParams = lp
        }

        val cfg = SessionConfiguration(
            SessionConfiguration.SESSION_REGULAR,
            listOf(OutputConfiguration(ps), OutputConfiguration(rd.surface)),
            handlerExecutor,
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(s: CameraCaptureSession) {
                    session = s
                    startPreview(dev, s, ps)
                    log("Ready: $mode ${capSize.width}x${capSize.height}, preview ${preview.width}x${preview.height}")
                }
                override fun onConfigureFailed(s: CameraCaptureSession) {
                    log("Session configuration failed for $mode ${capSize.width}x${capSize.height}")
                }
            }
        )
        dev.createCaptureSession(cfg)
    }

    private fun startPreview(dev: CameraDevice, s: CameraCaptureSession, ps: Surface) {
        val b = dev.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        b.addTarget(ps)
        b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        s.setRepeatingRequest(b.build(), object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(
                session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult
            ) { lastPreview = result }
        }, handler)
    }

    private fun <T> CaptureRequest.Builder.setIfAvailable(
        key: CaptureRequest.Key<T>, value: T, available: IntArray?, valueAsInt: Int
    ) {
        if (available != null && available.contains(valueAsInt)) set(key, value)
    }

    fun captureBurst() {
        val dev = device; val s = session; val rd = reader; val c = chars
        val prev = lastPreview
        if (dev == null || s == null || rd == null || c == null) { log("Camera not ready"); return }
        if (prev == null) { log("Waiting for preview metering, try again"); return }
        if (capturing) { log("Previous burst still saving"); return }

        // Freeze exposure from the live preview metering. Cap shutter at ~1/30 s for handheld; trade for ISO.
        var exp = prev.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: 10_000_000L
        var iso = prev.get(CaptureResult.SENSOR_SENSITIVITY) ?: 100
        val maxExp = 33_000_000L
        if (exp > maxExp) {
            iso = (iso.toLong() * exp / maxExp).toInt()
            exp = maxExp
        }
        val isoRange = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        if (isoRange != null) iso = iso.coerceIn(isoRange.lower, isoRange.upper)
        burstExposureNs = exp; burstIso = iso

        val b = dev.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)
        b.addTarget(rd.surface)
        b.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
        b.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
        b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exp)
        b.set(CaptureRequest.SENSOR_SENSITIVITY, iso)
        b.set(CaptureRequest.CONTROL_AWB_LOCK, true)
        b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        val nrModes = c.get(CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES)
        val edgeModes = c.get(CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES)
        if (mode == Mode.RAW) {
            b.setIfAvailable(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF, nrModes, CaptureRequest.NOISE_REDUCTION_MODE_OFF)
            b.setIfAvailable(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF, edgeModes, CaptureRequest.EDGE_MODE_OFF)
            b.set(CaptureRequest.STATISTICS_LENS_SHADING_MAP_MODE, CaptureRequest.STATISTICS_LENS_SHADING_MAP_MODE_ON)
        } else {
            b.set(CaptureRequest.JPEG_QUALITY, 100.toByte())
            b.setIfAvailable(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_MINIMAL, nrModes, CaptureRequest.NOISE_REDUCTION_MODE_MINIMAL)
        }

        results.clear(); images.clear(); failed = 0; lost = 0
        expected = burstCount
        capturing = true
        log("Capturing $expected frames: exposure %.1f ms, ISO %d …".format(exp / 1e6, iso))
        val requests = List(expected) { b.build() }
        s.captureBurst(requests, object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(
                session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult
            ) { results.add(result); maybeFinalize() }
            override fun onCaptureFailed(session: CameraCaptureSession, request: CaptureRequest, failure: CaptureFailure) {
                failed++; maybeFinalize()
            }
            override fun onCaptureBufferLost(
                session: CameraCaptureSession, request: CaptureRequest, target: Surface, frameNumber: Long
            ) { lost++; maybeFinalize() }
        }, handler)
        handler.postDelayed({ if (capturing) finalizeBurst(force = true) }, 20_000)
    }

    private fun maybeFinalize() {
        if (!capturing) return
        if (results.size + failed >= expected && images.size >= results.size - lost) finalizeBurst(force = false)
    }

    private fun finalizeBurst(force: Boolean) {
        capturing = false
        val c = chars ?: return
        val imgs = images.sortedBy { it.timestamp }
        val ress = results.sortedBy { it.get(CaptureResult.SENSOR_TIMESTAMP) ?: 0L }
        val n = minOf(imgs.size, ress.size)
        images.clear(); results.clear()
        if (force) log("Timed out; got ${imgs.size} images / ${ress.size} results, saving $n")
        if (n == 0) { imgs.forEach { it.close() }; log("No frames captured"); return }
        val folder = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + "_" + mode.name.lowercase()
        val m = mode; val size = imgs[0].let { "${it.width}x${it.height}" }
        log("Saving $n frames to Download/CamLab/$folder …")
        io.execute {
            try {
                val frames = JSONArray()
                for (i in 0 until n) {
                    val img = imgs[i]; val res = ress[i]
                    val name = "frame_%03d.%s".format(i, if (m == Mode.RAW) "dng" else "jpg")
                    if (m == Mode.RAW) {
                        Storage.open(ctx, folder, name, "image/x-adobe-dng").use { out ->
                            DngCreator(c, res).use { dng -> dng.writeImage(out, img) }
                        }
                    } else {
                        val buf = img.planes[0].buffer
                        val bytes = ByteArray(buf.remaining()); buf.get(bytes)
                        Storage.writeBytes(ctx, folder, name, "image/jpeg", bytes)
                    }
                    img.close()
                    frames.put(frameMeta(res).put("file", name))
                }
                for (i in n until imgs.size) imgs[i].close()
                val meta = JSONObject()
                    .put("mode", m.name)
                    .put("size", size)
                    .put("camera_id", cameraId)
                    .put("requested_exposure_ns", burstExposureNs)
                    .put("requested_iso", burstIso)
                    .put("device", JSONObject()
                        .put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL)
                        .put("android", Build.VERSION.RELEASE).put("build", Build.DISPLAY))
                    .put("static", staticMeta(c))
                    .put("frames", frames)
                Storage.writeText(ctx, folder, "meta.json", meta.toString(2), "application/json")
                log("Saved $n frames + meta.json to Download/CamLab/$folder")
            } catch (e: Exception) {
                log("Save failed: ${e.javaClass.simpleName}: ${e.message}")
                imgs.forEach { try { it.close() } catch (_: Exception) {} }
            }
        }
    }

    private fun rationals(arr: Array<Rational>?): JSONArray? =
        arr?.let { a -> JSONArray().also { j -> a.forEach { j.put(it.toDouble()) } } }

    private fun matrix(t: android.hardware.camera2.params.ColorSpaceTransform?): JSONArray? =
        t?.let {
            val j = JSONArray()
            for (row in 0 until 3) for (col in 0 until 3) j.put(it.getElement(col, row).toDouble())
            j
        }

    private fun staticMeta(c: CameraCharacteristics): JSONObject {
        val o = JSONObject()
        o.put("cfa", c.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT)) // 0 RGGB,1 GRBG,2 GBRG,3 BGGR
        o.put("white_level", c.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL))
        c.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)?.let { p ->
            // Order: (x0,y0) (x1,y0) (x0,y1) (x1,y1), i.e. row-major over the 2x2 CFA cell.
            o.put("black_level_pattern", JSONArray(listOf(
                p.getOffsetForIndex(0, 0), p.getOffsetForIndex(1, 0),
                p.getOffsetForIndex(0, 1), p.getOffsetForIndex(1, 1))))
        }
        o.put("sensor_orientation", c.get(CameraCharacteristics.SENSOR_ORIENTATION))
        c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)?.let { o.put("pixel_array", JSONArray(listOf(it.width, it.height))) }
        c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)?.let {
            o.put("active_array", JSONArray(listOf(it.left, it.top, it.right, it.bottom)))
        }
        o.put("illuminant1", c.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT1))
        o.put("illuminant2", c.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT2))
        o.put("color_matrix1", matrix(c.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM1)))
        o.put("color_matrix2", matrix(c.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM2)))
        o.put("forward_matrix1", matrix(c.get(CameraCharacteristics.SENSOR_FORWARD_MATRIX1)))
        o.put("forward_matrix2", matrix(c.get(CameraCharacteristics.SENSOR_FORWARD_MATRIX2)))
        return o
    }

    private fun frameMeta(r: TotalCaptureResult): JSONObject {
        val o = JSONObject()
        o.put("timestamp", r.get(CaptureResult.SENSOR_TIMESTAMP))
        o.put("exposure_ns", r.get(CaptureResult.SENSOR_EXPOSURE_TIME))
        o.put("iso", r.get(CaptureResult.SENSOR_SENSITIVITY))
        o.put("focus_distance", r.get(CaptureResult.LENS_FOCUS_DISTANCE)?.toDouble())
        o.put("neutral", rationals(r.get(CaptureResult.SENSOR_NEUTRAL_COLOR_POINT)))
        r.get(CaptureResult.SENSOR_NOISE_PROFILE)?.let { np ->
            val j = JSONArray()
            np.forEach { j.put(JSONArray(listOf(it.first, it.second))) }
            o.put("noise_profile", j) // per CFA channel (S, O): var = S*signal + O  (signal in [0,1] normalized)
        }
        r.get(CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL)?.let { d ->
            o.put("dynamic_black", JSONArray(d.map { it.toDouble() }))
        }
        o.put("dynamic_white", r.get(CaptureResult.SENSOR_DYNAMIC_WHITE_LEVEL))
        r.get(CaptureResult.STATISTICS_LENS_SHADING_CORRECTION_MAP)?.let { lsm ->
            val cols = lsm.columnCount; val rows = lsm.rowCount
            val g = JSONArray()
            for (y in 0 until rows) for (x in 0 until cols) for (ch in 0 until 4) g.put(lsm.getGainFactor(ch, x, y).toDouble())
            o.put("shading", JSONObject().put("cols", cols).put("rows", rows).put("gains_row_major_rggb", g))
        }
        return o
    }

    fun close() {
        capturing = false
        try { session?.close() } catch (_: Exception) {}
        try { device?.close() } catch (_: Exception) {}
        try { reader?.close() } catch (_: Exception) {}
        previewSurface?.release()
        session = null; device = null; reader = null; previewSurface = null
        lastPreview = null
        results.clear(); images.forEach { try { it.close() } catch (_: Exception) {} }; images.clear()
    }

    fun shutdown() {
        close()
        thread.quitSafely()
        io.shutdown()
    }
}
