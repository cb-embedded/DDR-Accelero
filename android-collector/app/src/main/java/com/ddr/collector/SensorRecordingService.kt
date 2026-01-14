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
import co.nstant.`in`.cbor.CborBuilder
import co.nstant.`in`.cbor.CborEncoder
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*

class SensorRecordingService : Service(), SensorEventListener {
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private val sensorData = mutableListOf<SensorSample>()
    private var lastFramerateUpdate = 0L
    private var frameCount = 0
    private var currentFramerate = 0f
    private var wakeLock: PowerManager.WakeLock? = null
    private var isRecording = false
    private var lastFile: File? = null
    
    private val binder = LocalBinder()
    private var framerateCallback: ((Float) -> Unit)? = null

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
        
        sensorData.clear()
        frameCount = 0
        lastFramerateUpdate = System.currentTimeMillis()
        
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
        
        if (sensorData.isNotEmpty()) {
            saveDataToCbor()
        }
        
        return lastFile
    }

    fun isRecording(): Boolean = isRecording

    fun setFramerateCallback(callback: (Float) -> Unit) {
        framerateCallback = callback
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!isRecording) return

        val timestamp = event.timestamp

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                sensorData.add(SensorSample(
                    timestamp,
                    event.values[0], event.values[1], event.values[2],
                    null, null, null
                ))
            }
            Sensor.TYPE_GYROSCOPE -> {
                sensorData.add(SensorSample(
                    timestamp,
                    null, null, null,
                    event.values[0], event.values[1], event.values[2]
                ))
            }
        }

        frameCount++
        val now = System.currentTimeMillis()
        if (now - lastFramerateUpdate >= 1000) {
            currentFramerate = frameCount / ((now - lastFramerateUpdate) / 1000f)
            framerateCallback?.invoke(currentFramerate)
            
            updateNotification("Recording: %.1f Hz".format(currentFramerate))
            
            frameCount = 0
            lastFramerateUpdate = now
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun saveDataToCbor() {
        val timestamp = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.US).format(Date())
        val filename = "sensor_data_$timestamp.cbor"
        val file = File(filesDir, filename)
        lastFile = file

        val accelSamples = sensorData.filter { it.accelX != null }
        val gyroSamples = sensorData.filter { it.gyroX != null }

        FileOutputStream(file).use { fos ->
            val builder = CborBuilder()
            val map = builder.addMap()
            
            map.put("device", Build.MODEL)
            map.put("timestamp", timestamp)
            map.put("accelerometer_count", accelSamples.size.toLong())
            map.put("gyroscope_count", gyroSamples.size.toLong())
            
            val accelArray = map.putArray("accelerometer")
            accelSamples.forEach { sample ->
                val sampleMap = accelArray.addMap()
                sampleMap.put("timestamp", sample.timestamp)
                sampleMap.put("x", sample.accelX!!.toDouble())
                sampleMap.put("y", sample.accelY!!.toDouble())
                sampleMap.put("z", sample.accelZ!!.toDouble())
                sampleMap.end()
            }
            accelArray.end()
            
            val gyroArray = map.putArray("gyroscope")
            gyroSamples.forEach { sample ->
                val sampleMap = gyroArray.addMap()
                sampleMap.put("timestamp", sample.timestamp)
                sampleMap.put("x", sample.gyroX!!.toDouble())
                sampleMap.put("y", sample.gyroY!!.toDouble())
                sampleMap.put("z", sample.gyroZ!!.toDouble())
                sampleMap.end()
            }
            gyroArray.end()
            
            map.end()
            
            CborEncoder(fos).encode(builder.build())
        }
    }

    fun getLastFile(): File? = lastFile

    companion object {
        private const val CHANNEL_ID = "SensorRecordingChannel"
        private const val NOTIFICATION_ID = 1
    }
}
