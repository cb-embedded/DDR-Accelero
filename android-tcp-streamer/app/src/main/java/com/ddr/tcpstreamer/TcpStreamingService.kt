package com.ddr.tcpstreamer

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
import java.io.PrintWriter
import java.net.Socket

class TcpStreamingService : Service(), SensorEventListener {
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var socket: Socket? = null
    private var writer: PrintWriter? = null
    private var accelFrameCount = 0
    private var gyroFrameCount = 0
    private var lastFramerateUpdate = 0L
    private var currentAccelFramerate = 0f
    private var currentGyroFramerate = 0f
    private var wakeLock: PowerManager.WakeLock? = null
    private var isStreaming = false
    private var serverIpAddress = ""
    private var serverPort = 5000
    
    private val binder = LocalBinder()
    private var framerateCallback: ((Float, Float) -> Unit)? = null
    private var statusCallback: ((String) -> Unit)? = null

    inner class LocalBinder : Binder() {
        fun getService(): TcpStreamingService = this@TcpStreamingService
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
            "DDRTcpStreamer::SensorStreamingWakeLock"
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        createNotificationChannel()
        
        val notification = createNotification("Connecting...")
        startForeground(NOTIFICATION_ID, notification)
        
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        stopStreaming()
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Sensor Streaming",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Notification for sensor data streaming"
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
            .setContentTitle("DDR TCP Streamer")
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

    fun startStreaming(ipAddress: String, port: Int) {
        if (isStreaming) return
        
        serverIpAddress = ipAddress
        serverPort = port
        accelFrameCount = 0
        gyroFrameCount = 0
        lastFramerateUpdate = System.currentTimeMillis()
        
        Thread {
            try {
                updateNotification("Connecting to $ipAddress:$port...")
                statusCallback?.invoke("Connecting...")
                
                socket = Socket(ipAddress, port)
                writer = PrintWriter(socket!!.getOutputStream(), true)
                
                updateNotification("Connected to $ipAddress:$port")
                statusCallback?.invoke("Connected")
                
                wakeLock?.acquire(WAKE_LOCK_TIMEOUT_MS)
                
                accelerometer?.let {
                    sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
                }
                gyroscope?.let {
                    sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
                }
                
                isStreaming = true
                
            } catch (e: Exception) {
                e.printStackTrace()
                updateNotification("Connection failed: ${e.message}")
                statusCallback?.invoke("Connection failed: ${e.message}")
                stopStreaming()
            }
        }.start()
    }

    fun stopStreaming() {
        if (!isStreaming) return
        
        sensorManager.unregisterListener(this)
        isStreaming = false
        
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
        
        try {
            writer?.close()
            socket?.close()
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            writer = null
            socket = null
        }
        
        statusCallback?.invoke("Disconnected")
    }

    fun isStreaming(): Boolean = isStreaming

    fun setFramerateCallback(callback: (Float, Float) -> Unit) {
        framerateCallback = callback
    }

    fun setStatusCallback(callback: (String) -> Unit) {
        statusCallback = callback
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!isStreaming || writer == null) return

        val timestampNs = event.timestamp
        val timestampMs = System.currentTimeMillis()

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                try {
                    writer?.println("$timestampNs,$timestampMs,accel,${event.values[0]},${event.values[1]},${event.values[2]}")
                    accelFrameCount++
                } catch (e: Exception) {
                    e.printStackTrace()
                    stopStreaming()
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                try {
                    writer?.println("$timestampNs,$timestampMs,gyro,${event.values[0]},${event.values[1]},${event.values[2]}")
                    gyroFrameCount++
                } catch (e: Exception) {
                    e.printStackTrace()
                    stopStreaming()
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

    companion object {
        private const val CHANNEL_ID = "SensorStreamingChannel"
        private const val NOTIFICATION_ID = 1
        private const val WAKE_LOCK_TIMEOUT_MS = 60 * 60 * 1000L // 1 hour
    }
}
