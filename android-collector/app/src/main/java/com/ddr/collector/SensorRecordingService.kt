package com.ddr.collector

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*

class SensorRecordingService : Service(), SensorEventListener {
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private val accelSampleBuffer = mutableListOf<SensorSample>()
    private val gyroSampleBuffer = mutableListOf<SensorSample>()
    private var currentFile: File? = null
    private var fileOutputStream: BufferedOutputStream? = null
    private var binaryWriter: BinaryFormatWriter? = null
    private var accelSampleCount = 0L
    private var gyroSampleCount = 0L
    private var lastFramerateUpdate = 0L
    private var accelFrameCount = 0
    private var gyroFrameCount = 0
    private var currentAccelFramerate = 0f
    private var currentGyroFramerate = 0f
    private var wakeLock: PowerManager.WakeLock? = null
    private var isRecording = false
    private var lastFile: File? = null
    
    private val binder = LocalBinder()
    private var framerateCallback: ((Float, Float) -> Unit)? = null

    data class SensorSample(
        val timestamp: Long,
        val accelX: Float?,
        val accelY: Float?,
        val accelZ: Float?,
        val gyroX: Float?,
        val gyroY: Float?,
        val gyroZ: Float?
    )

    inner class LocalBinder : Binder() {
        fun getService(): SensorRecordingService = this@SensorRecordingService
    }

    override fun onBind(intent: Intent): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        
        val powerManager = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "DDRCollector::SensorRecordingWakeLock"
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        createNotificationChannel()
        
        val notification = createNotification("Recording sensor data...")
        startForeground(NOTIFICATION_ID, notification)
        
        startRecording()
        
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        stopRecording()
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Sensor Recording",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Notification for sensor data recording"
                setShowBadge(false)
            }
            
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(contentText: String): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("DDR Collector")
            .setContentText(contentText)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(contentText: String) {
        val notification = createNotification(contentText)
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, notification)
    }

    fun startRecording() {
        if (isRecording) return
        
        accelSampleBuffer.clear()
        gyroSampleBuffer.clear()
        accelSampleCount = 0L
        gyroSampleCount = 0L
        accelFrameCount = 0
        gyroFrameCount = 0
        lastFramerateUpdate = System.currentTimeMillis()
        
        // Create file and start writing
        val timestamp = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.US).format(Date())
        val filename = "sensor_data_$timestamp.ddrbin"
        currentFile = File(filesDir, filename)
        lastFile = currentFile
        
        try {
            fileOutputStream = BufferedOutputStream(FileOutputStream(currentFile!!))
            binaryWriter = BinaryFormatWriter(fileOutputStream!!)
            
            // Write header with current timestamp in nanoseconds
            val timestampNs = System.currentTimeMillis() * 1_000_000
            binaryWriter?.writeHeader(timestampNs)
        } catch (e: Exception) {
            e.printStackTrace()
            android.util.Log.e("SensorRecordingService", "Failed to create recording file", e)
            currentFile = null
            lastFile = null
            binaryWriter = null
            return
        }
        
        wakeLock?.acquire(60 * 60 * 1000L) // 1 hour timeout
        
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        }
        
        isRecording = true
    }

    fun stopRecording(): File? {
        if (!isRecording) return lastFile
        
        sensorManager.unregisterListener(this)
        isRecording = false
        
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
        
        // Flush remaining samples
        flushAccelBuffer()
        flushGyroBuffer()
        
        // Close file
        try {
            binaryWriter?.flush()
            binaryWriter?.close()
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            binaryWriter = null
            try {
                fileOutputStream?.close()
            } catch (e: Exception) {
                e.printStackTrace()
            }
            fileOutputStream = null
            currentFile = null
        }
        
        return lastFile
    }

    fun isRecording(): Boolean = isRecording

    fun setFramerateCallback(callback: (Float, Float) -> Unit) {
        framerateCallback = callback
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!isRecording) return

        val timestamp = event.timestamp

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accelSampleBuffer.add(SensorSample(
                    timestamp,
                    event.values[0], event.values[1], event.values[2],
                    null, null, null
                ))
                accelFrameCount++
                accelSampleCount++
                
                if (accelSampleBuffer.size >= BUFFER_SIZE) {
                    flushAccelBuffer()
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyroSampleBuffer.add(SensorSample(
                    timestamp,
                    null, null, null,
                    event.values[0], event.values[1], event.values[2]
                ))
                gyroFrameCount++
                gyroSampleCount++
                
                if (gyroSampleBuffer.size >= BUFFER_SIZE) {
                    flushGyroBuffer()
                }
            }
        }

        val now = System.currentTimeMillis()
        if (now - lastFramerateUpdate >= 1000) {
            val elapsedSeconds = (now - lastFramerateUpdate) / 1000f
            currentAccelFramerate = accelFrameCount / elapsedSeconds
            currentGyroFramerate = gyroFrameCount / elapsedSeconds
            framerateCallback?.invoke(currentAccelFramerate, currentGyroFramerate)
            
            updateNotification("Accel: %.0f Hz, Gyro: %.0f Hz".format(currentAccelFramerate, currentGyroFramerate))
            
            accelFrameCount = 0
            gyroFrameCount = 0
            lastFramerateUpdate = now
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    
    private fun flushAccelBuffer() {
        if (accelSampleBuffer.isEmpty()) return
        
        try {
            accelSampleBuffer.forEach { sample ->
                binaryWriter?.writeRecord(
                    BinaryFormatWriter.SENSOR_TYPE_ACCELEROMETER,
                    sample.timestamp,
                    sample.accelX!!,
                    sample.accelY!!,
                    sample.accelZ!!
                )
            }
            
            accelSampleBuffer.clear()
        } catch (e: Exception) {
            e.printStackTrace()
            android.util.Log.e("SensorRecordingService", "Failed to flush accelerometer buffer", e)
        }
    }
    
    private fun flushGyroBuffer() {
        if (gyroSampleBuffer.isEmpty()) return
        
        try {
            gyroSampleBuffer.forEach { sample ->
                binaryWriter?.writeRecord(
                    BinaryFormatWriter.SENSOR_TYPE_GYROSCOPE,
                    sample.timestamp,
                    sample.gyroX!!,
                    sample.gyroY!!,
                    sample.gyroZ!!
                )
            }
            
            gyroSampleBuffer.clear()
        } catch (e: Exception) {
            e.printStackTrace()
            android.util.Log.e("SensorRecordingService", "Failed to flush gyroscope buffer", e)
        }
    }

    fun getLastFile(): File? = lastFile

    companion object {
        private const val CHANNEL_ID = "SensorRecordingChannel"
        private const val NOTIFICATION_ID = 1
        private const val BUFFER_SIZE = 100 // Flush to disk every 100 samples
    }
}
