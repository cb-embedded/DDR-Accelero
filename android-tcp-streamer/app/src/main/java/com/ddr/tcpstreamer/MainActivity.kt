package com.ddr.tcpstreamer

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
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {
    private var streamingService: TcpStreamingService? = null
    private var serviceBound = false
    private lateinit var ipAddressInput: EditText
    private lateinit var portInput: EditText
    private lateinit var framerateText: TextView
    private lateinit var statusText: TextView
    private lateinit var toggleButton: Button

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val localBinder = binder as TcpStreamingService.LocalBinder
            streamingService = localBinder.getService()
            serviceBound = true
            
            streamingService?.setFramerateCallback { accelFramerate, gyroFramerate ->
                runOnUiThread {
                    framerateText.text = "Accel: %.0f Hz, Gyro: %.0f Hz".format(accelFramerate, gyroFramerate)
                }
            }
            
            streamingService?.setStatusCallback { status ->
                runOnUiThread {
                    statusText.text = "Status: $status"
                }
            }
            
            updateUIState()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            streamingService = null
            serviceBound = false
        }
    }

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (!isGranted) {
            Toast.makeText(this, 
                "Notification permission needed for background streaming", 
                Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ipAddressInput = findViewById(R.id.ipAddressInput)
        portInput = findViewById(R.id.portInput)
        framerateText = findViewById(R.id.framerateText)
        statusText = findViewById(R.id.statusText)
        toggleButton = findViewById(R.id.toggleButton)

        portInput.setText("5000")

        checkPermissions()
        checkBatteryOptimization()

        toggleButton.setOnClickListener {
            val isStreaming = streamingService?.isStreaming() == true
            if (isStreaming) stopStreaming() else startStreaming()
        }
    }

    override fun onStart() {
        super.onStart()
        val intent = Intent(this, TcpStreamingService::class.java)
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
                    .setMessage("To maintain consistent sensor sampling rates during background streaming, " +
                               "this app needs to be exempted from battery optimization.")
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

    private fun startStreaming() {
        val ipAddress = ipAddressInput.text.toString().trim()
        val portStr = portInput.text.toString().trim()
        
        if (ipAddress.isEmpty()) {
            Toast.makeText(this, "Please enter IP address", Toast.LENGTH_SHORT).show()
            return
        }
        
        val port = portStr.toIntOrNull() ?: 5000
        
        val intent = Intent(this, TcpStreamingService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
        
        if (!serviceBound) {
            bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        }
        
        streamingService?.startStreaming(ipAddress, port)
        
        updateUIState()
    }

    private fun stopStreaming() {
        streamingService?.stopStreaming()
        stopService(Intent(this, TcpStreamingService::class.java))
        updateUIState()
    }

    private fun updateUIState() {
        val isStreaming = streamingService?.isStreaming() == true
        toggleButton.text = if (isStreaming) "Stop Streaming" else "Start Streaming"
        ipAddressInput.isEnabled = !isStreaming
        portInput.isEnabled = !isStreaming
    }

    companion object {
        private const val PERMISSION_REQUEST_CODE = 1
    }
}
