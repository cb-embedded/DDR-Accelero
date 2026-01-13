package com.ddr.collector

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import co.nstant.in.cbor.CborBuilder
import co.nstant.in.cbor.CborEncoder
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity(), SensorEventListener {
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var isRecording = false
    private val accelData = mutableListOf<FloatArray>()
    private val gyroData = mutableListOf<FloatArray>()
    private val timestamps = mutableListOf<Long>()
    private var lastFramerateUpdate = 0L
    private var frameCount = 0
    private var currentFramerate = 0f
    private lateinit var framerateText: TextView
    private lateinit var toggleButton: Button
    private lateinit var downloadButton: Button
    private lateinit var shareButton: Button
    private var lastFile: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        framerateText = findViewById(R.id.framerateText)
        toggleButton = findViewById(R.id.toggleButton)
        downloadButton = findViewById(R.id.downloadButton)
        shareButton = findViewById(R.id.shareButton)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            checkPermissions()
        }

        toggleButton.setOnClickListener {
            if (isRecording) stopRecording() else startRecording()
        }

        downloadButton.setOnClickListener { downloadFile() }
        shareButton.setOnClickListener { shareFile() }
    }

    private fun checkPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.HIGH_SAMPLING_RATE_SENSORS) 
                != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, 
                    arrayOf(Manifest.permission.HIGH_SAMPLING_RATE_SENSORS), 1)
            }
        }
    }

    private fun startRecording() {
        accelData.clear()
        gyroData.clear()
        timestamps.clear()
        frameCount = 0
        lastFramerateUpdate = System.currentTimeMillis()

        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        }

        isRecording = true
        toggleButton.text = "Stop Recording"
        downloadButton.isEnabled = false
        shareButton.isEnabled = false
    }

    private fun stopRecording() {
        sensorManager.unregisterListener(this)
        isRecording = false
        toggleButton.text = "Start Recording"

        if (timestamps.isNotEmpty()) {
            saveDataToCbor()
            downloadButton.isEnabled = true
            shareButton.isEnabled = true
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!isRecording) return

        val timestamp = System.nanoTime()
        timestamps.add(timestamp)

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accelData.add(floatArrayOf(event.values[0], event.values[1], event.values[2]))
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyroData.add(floatArrayOf(event.values[0], event.values[1], event.values[2]))
            }
        }

        frameCount++
        val now = System.currentTimeMillis()
        if (now - lastFramerateUpdate >= 1000) {
            currentFramerate = frameCount / ((now - lastFramerateUpdate) / 1000f)
            runOnUiThread {
                framerateText.text = "Framerate: %.1f Hz".format(currentFramerate)
            }
            frameCount = 0
            lastFramerateUpdate = now
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun saveDataToCbor() {
        val timestamp = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.US).format(Date())
        val filename = "sensor_data_$timestamp.cbor"
        lastFile = File(filesDir, filename)

        FileOutputStream(lastFile!!).use { fos ->
            val builder = CborBuilder()
            val map = builder.addMap()
            
            map.put("device", Build.MODEL)
            map.put("timestamp", timestamp)
            map.put("sample_count", timestamps.size.toLong())
            
            val timestampArray = map.putArray("timestamps")
            timestamps.forEach { timestampArray.add(it) }
            
            val accelArray = map.putArray("accelerometer")
            accelData.forEach { sample ->
                val sampleMap = accelArray.addMap()
                sampleMap.put("x", sample[0].toDouble())
                sampleMap.put("y", sample[1].toDouble())
                sampleMap.put("z", sample[2].toDouble())
                sampleMap.end()
            }
            accelArray.end()
            
            val gyroArray = map.putArray("gyroscope")
            gyroData.forEach { sample ->
                val sampleMap = gyroArray.addMap()
                sampleMap.put("x", sample[0].toDouble())
                sampleMap.put("y", sample[1].toDouble())
                sampleMap.put("z", sample[2].toDouble())
                sampleMap.end()
            }
            gyroArray.end()
            
            map.end()
            
            CborEncoder(fos).encode(builder.build())
        }

        Toast.makeText(this, "Saved: $filename", Toast.LENGTH_SHORT).show()
    }

    private fun downloadFile() {
        lastFile?.let { file ->
            val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "application/cbor"
                putExtra(Intent.EXTRA_TITLE, file.name)
            }
            startActivityForResult(intent, REQUEST_SAVE_FILE)
        }
    }

    private fun shareFile() {
        lastFile?.let { file ->
            val uri = FileProvider.getUriForFile(this, 
                "${applicationContext.packageName}.provider", file)
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "application/cbor"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(intent, "Share sensor data"))
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_SAVE_FILE && resultCode == RESULT_OK) {
            data?.data?.let { uri ->
                lastFile?.inputStream()?.use { input ->
                    contentResolver.openOutputStream(uri)?.use { output ->
                        input.copyTo(output)
                        Toast.makeText(this, "File saved", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    companion object {
        private const val REQUEST_SAVE_FILE = 1
    }
}
