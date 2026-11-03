package dev.camlab

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.util.TypedValue
import android.view.Gravity
import android.view.TextureView
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

class MainActivity : Activity() {

    private lateinit var texture: TextureView
    private lateinit var logView: TextView
    private lateinit var scroll: ScrollView
    private lateinit var modeBtn: Button
    private lateinit var countLabel: TextView
    private var capture: BurstCapture? = null
    private var mode = BurstCapture.Mode.RAW
    private var count = 6

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val dm = resources.displayMetrics
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }

        texture = TextureView(this)
        root.addView(texture, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dm.widthPixels * 4 / 3))

        fun row() = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        fun btn(text: String, onClick: () -> Unit) = Button(this).apply {
            this.text = text; isAllCaps = false; setOnClickListener { onClick() }
        }
        val w1 = { LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f) }

        val row1 = row()
        row1.addView(btn("Probe device") { runProbe() }, w1())
        modeBtn = btn("Mode: RAW") { toggleMode() }
        row1.addView(modeBtn, w1())
        root.addView(row1)

        val row2 = row()
        row2.addView(btn("−") { setCount(count - 1) }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 0.5f))
        countLabel = TextView(this).apply { gravity = Gravity.CENTER; text = "6 frames" }
        row2.addView(countLabel, w1())
        row2.addView(btn("+") { setCount(count + 1) }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 0.5f))
        row2.addView(btn("CAPTURE BURST") { capture?.captureBurst() ?: log("Camera not open") }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.5f))
        root.addView(row2)

        logView = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 10f)
            setTextIsSelectable(true)
            setPadding(16, 8, 16, 8)
        }
        scroll = ScrollView(this).apply { addView(logView) }
        root.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        setContentView(root)

        log("CamLab ${Build.MANUFACTURER} ${Build.MODEL}, Android ${Build.VERSION.RELEASE}")
        log("Tap 'Probe device' first. Files are saved to Download/CamLab/.")
    }

    override fun onResume() {
        super.onResume()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
        } else {
            startCamera()
        }
    }

    override fun onPause() {
        capture?.shutdown()
        capture = null
        super.onPause()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) startCamera()
        else log("Camera permission denied")
    }

    private fun startCamera() {
        if (capture != null) return
        val start = {
            val c = BurstCapture(this, texture) { log(it) }
            c.burstCount = count
            capture = c
            c.open(mode)
        }
        if (texture.isAvailable) {
            start()
        } else {
            texture.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                override fun onSurfaceTextureAvailable(s: SurfaceTexture, w: Int, h: Int) { start() }
                override fun onSurfaceTextureSizeChanged(s: SurfaceTexture, w: Int, h: Int) {}
                override fun onSurfaceTextureDestroyed(s: SurfaceTexture) = true
                override fun onSurfaceTextureUpdated(s: SurfaceTexture) {}
            }
        }
    }

    private fun toggleMode() {
        mode = if (mode == BurstCapture.Mode.RAW) BurstCapture.Mode.JPEG else BurstCapture.Mode.RAW
        modeBtn.text = "Mode: ${mode.name}"
        capture?.open(mode)
    }

    private fun setCount(n: Int) {
        count = n.coerceIn(2, 12)
        countLabel.text = "$count frames"
        capture?.let { it.burstCount = count; it.open(mode) } // reader buffers depend on burst size
    }

    private fun runProbe() {
        val report = try { CapabilityProbe.report(this) } catch (e: Exception) { "Probe failed: $e" }
        log(report)
        val file = "probe_${Build.MODEL.replace(Regex("[^A-Za-z0-9]"), "_")}.txt"
        CapabilityProbe.probeCameraXExtensions(this) { line ->
            log(line)
            try {
                Storage.writeText(this, "probe", file, report + "\n" + line + "\n")
                log("Report saved to Download/CamLab/probe/$file")
            } catch (e: Exception) {
                log("Could not save report: ${e.message}")
            }
        }
    }

    private fun log(msg: String) {
        runOnUiThread {
            logView.append(msg + "\n")
            scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }
}
