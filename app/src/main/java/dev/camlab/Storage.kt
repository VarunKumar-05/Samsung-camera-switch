package dev.camlab

import android.content.ContentValues
import android.content.Context
import android.os.Environment
import android.provider.MediaStore
import java.io.OutputStream

/** Writes into Download/CamLab/<folder>/ via MediaStore (no storage permission needed, visible in My Files / USB). */
object Storage {
    fun open(ctx: Context, folder: String, name: String, mime: String): OutputStream {
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, mime)
            put(MediaStore.MediaColumns.RELATIVE_PATH, "${Environment.DIRECTORY_DOWNLOADS}/CamLab/$folder")
        }
        val uri = ctx.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: throw IllegalStateException("MediaStore insert failed for $name")
        return ctx.contentResolver.openOutputStream(uri)
            ?: throw IllegalStateException("openOutputStream failed for $name")
    }

    fun writeBytes(ctx: Context, folder: String, name: String, mime: String, bytes: ByteArray) {
        open(ctx, folder, name, mime).use { it.write(bytes) }
    }

    fun writeText(ctx: Context, folder: String, name: String, text: String, mime: String = "text/plain") {
        writeBytes(ctx, folder, name, mime, text.toByteArray())
    }
}
