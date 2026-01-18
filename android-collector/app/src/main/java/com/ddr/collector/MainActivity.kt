package com.ddr.collector

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.os.PowerManager
import android.provider.Settings
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File

class MainActivity : AppCompatActivity() {
    private var sensorService: SensorRecordingService? = null
    private var serviceBound = false
    private lateinit var framerateText: TextView
    private lateinit var toggleButton: Button
    private lateinit var downloadButton: Button
    private lateinit var shareButton: Button
    private lateinit var manageFilesButton: Button
    private var lastFile: File? = null

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val localBinder = binder as SensorRecordingService.LocalBinder
            sensorService = localBinder.getService()
            serviceBound = true
            
            sensorService?.setFramerateCallback { accelFramerate, gyroFramerate ->
                runOnUiThread {
                    framerateText.text = "Accel: %.0f Hz, Gyro: %.0f Hz".format(accelFramerate, gyroFramerate)
                }
            }
            
            updateUIState()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            sensorService = null
            serviceBound = false
        }
    }

    private val saveFileLauncher = registerForActivityResult(
        ActivityResultContracts.CreateDocument("application/octet-stream")
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

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (!isGranted) {
            Toast.makeText(this, 
                "Notification permission needed for background recording", 
                Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        framerateText = findViewById(R.id.framerateText)
        toggleButton = findViewById(R.id.toggleButton)
        downloadButton = findViewById(R.id.downloadButton)
        shareButton = findViewById(R.id.shareButton)
        manageFilesButton = findViewById(R.id.manageFilesButton)

        checkPermissions()

        checkBatteryOptimization()

        toggleButton.setOnClickListener {
            val isRecording = sensorService?.isRecording() == true
            if (isRecording) stopRecording() else startRecording()
        }

        downloadButton.setOnClickListener { downloadFile() }
        shareButton.setOnClickListener { shareFile() }
        manageFilesButton.setOnClickListener { 
            startActivity(Intent(this, FileManagerActivity::class.java))
        }
    }

    override fun onStart() {
        super.onStart()
        val intent = Intent(this, SensorRecordingService::class.java)
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    override fun onStop() {
        super.onStop()
        if (serviceBound) {
            unbindService(serviceConnection)
            serviceBound = false
        }
    }

    private fun checkPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.HIGH_SAMPLING_RATE_SENSORS)
                != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                    arrayOf(Manifest.permission.HIGH_SAMPLING_RATE_SENSORS), PERMISSION_REQUEST_CODE)
            }
        }
    }

    private fun checkBatteryOptimization() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
            val packageName = packageName
            
            if (!powerManager.isIgnoringBatteryOptimizations(packageName)) {
                AlertDialog.Builder(this)
                    .setTitle("Battery Optimization")
                    .setMessage("To maintain consistent sensor sampling rates during background recording, " +
                               "this app needs to be exempted from battery optimization. " +
                               "Without this exemption, the system may throttle the app and reduce data quality.")
                    .setPositiveButton("Grant") { _, _ ->
                        val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                            data = Uri.parse("package:$packageName")
                        }
                        try {
                            startActivity(intent)
                        } catch (e: Exception) {
                            Toast.makeText(this, "Unable to open battery settings", Toast.LENGTH_SHORT).show()
                        }
                    }
                    .setNegativeButton("Later", null)
                    .show()
            }
        }
    }

    private fun startRecording() {
        val intent = Intent(this, SensorRecordingService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
        
        if (!serviceBound) {
            bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        }
        
        updateUIState()
    }

    private fun stopRecording() {
        sensorService?.let { service ->
            lastFile = service.stopRecording()
            
            if (lastFile != null) {
                Toast.makeText(this, "Saved: ${lastFile?.name}", Toast.LENGTH_SHORT).show()
                downloadButton.isEnabled = true
                shareButton.isEnabled = true
            }
        }
        
        stopService(Intent(this, SensorRecordingService::class.java))
        
        updateUIState()
    }

    private fun updateUIState() {
        val isRecording = sensorService?.isRecording() == true
        toggleButton.text = if (isRecording) "Stop Recording" else "Start Recording"
        downloadButton.isEnabled = !isRecording && lastFile != null
        shareButton.isEnabled = !isRecording && lastFile != null
        
        if (!isRecording) {
            lastFile = sensorService?.getLastFile()
            if (lastFile != null) {
                downloadButton.isEnabled = true
                shareButton.isEnabled = true
            }
        }
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
                type = "application/octet-stream"
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
