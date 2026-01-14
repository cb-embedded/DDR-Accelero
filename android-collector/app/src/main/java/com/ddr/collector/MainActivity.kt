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
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import co.nstant.`in`.cbor.CborBuilder
import co.nstant.`in`.cbor.CborEncoder
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity(), SensorEventListener {
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var isRecording = false
    private val sensorData = mutableListOf<SensorSample>()
    private var lastFramerateUpdate = 0L
    private var frameCount = 0
    private var currentFramerate = 0f
    private lateinit var framerateText: TextView
    private lateinit var toggleButton: Button
    private lateinit var downloadButton: Button
    private lateinit var shareButton: Button
    private var lastFile: File? = null

    data class SensorSample(
        val timestamp: Long,
        val accelX: Float?,
        val accelY: Float?,
        val accelZ: Float?,
        val gyroX: Float?,
        val gyroY: Float?,
        val gyroZ: Float?
    )

    private val saveFileLauncher = registerForActivityResult(
        ActivityResultContracts.CreateDocument("application/cbor")
    ) { uri ->
        uri?.let {
            lastFile?.inputStream()?.use { input ->
                contentResolver.openOutputStream(it)?.use { output ->
                    input.copyTo(output)
                    Toast.makeText(this, "File saved", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

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

    override fun onPause() {
        super.onPause()
        if (isRecording) {
            sensorManager.unregisterListener(this)
        }
    }

    override fun onResume() {
        super.onResume()
        if (isRecording) {
            accelerometer?.let {
                sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
            }
            gyroscope?.let {
                sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
            }
        }
    }

    private fun checkPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.HIGH_SAMPLING_RATE_SENSORS)
                != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                    arrayOf(Manifest.permission.HIGH_SAMPLING_RATE_SENSORS), PERMISSION_REQUEST_CODE)
            }
        }
    }

    private fun startRecording() {
        sensorData.clear()
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

        if (sensorData.isNotEmpty()) {
            saveDataToCbor()
            downloadButton.isEnabled = true
            shareButton.isEnabled = true
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!isRecording) return

        val timestamp = System.nanoTime()

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

        Toast.makeText(this, "Saved: $filename", Toast.LENGTH_SHORT).show()
    }

    private fun downloadFile() {
        lastFile?.let { file ->
            saveFileLauncher.launch(file.name)
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

    companion object {
        private const val PERMISSION_REQUEST_CODE = 1
    }
}
