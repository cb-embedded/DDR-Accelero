package com.ddr.collector

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

class FileManagerActivity : AppCompatActivity() {
    private lateinit var recyclerView: RecyclerView
    private lateinit var emptyView: TextView
    private lateinit var adapter: FileAdapter
    private val files = mutableListOf<File>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_file_manager)

        recyclerView = findViewById(R.id.fileRecyclerView)
        emptyView = findViewById(R.id.emptyView)

        recyclerView.layoutManager = LinearLayoutManager(this)
        adapter = FileAdapter(files,
            onShare = { shareFile(it) },
            onDownload = { downloadFile(it) },
            onDelete = { deleteFile(it) }
        )
        recyclerView.adapter = adapter

        loadFiles()
    }

    override fun onResume() {
        super.onResume()
        loadFiles()
    }

    private fun loadFiles() {
        files.clear()
        filesDir.listFiles { file ->
            file.name.endsWith(".ddrbin")
        }?.sortedByDescending { it.lastModified() }?.let {
            files.addAll(it)
        }
        
        adapter.notifyDataSetChanged()
        
        if (files.isEmpty()) {
            recyclerView.visibility = View.GONE
            emptyView.visibility = View.VISIBLE
        } else {
            recyclerView.visibility = View.VISIBLE
            emptyView.visibility = View.GONE
        }
    }

    private val saveFileLauncher = registerForActivityResult(
        ActivityResultContracts.CreateDocument("application/octet-stream")
    ) { uri ->
        uri?.let {
            adapter.pendingDownloadFile?.inputStream()?.use { input ->
                contentResolver.openOutputStream(it)?.use { output ->
                    input.copyTo(output)
                    Toast.makeText(this, "File saved", Toast.LENGTH_SHORT).show()
                }
            }
            adapter.pendingDownloadFile = null
        }
    }

    private fun shareFile(file: File) {
        val uri = FileProvider.getUriForFile(this,
            "${applicationContext.packageName}.provider", file)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "application/octet-stream"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, "Share sensor data"))
    }

    private fun downloadFile(file: File) {
        adapter.pendingDownloadFile = file
        saveFileLauncher.launch(file.name)
    }

    private fun deleteFile(file: File) {
        AlertDialog.Builder(this)
            .setTitle("Delete File")
            .setMessage("Are you sure you want to delete ${file.name}?")
            .setPositiveButton("Delete") { _, _ ->
                if (file.delete()) {
                    Toast.makeText(this, "File deleted", Toast.LENGTH_SHORT).show()
                    loadFiles()
                } else {
                    Toast.makeText(this, "Failed to delete file", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    class FileAdapter(
        private val files: List<File>,
        private val onShare: (File) -> Unit,
        private val onDownload: (File) -> Unit,
        private val onDelete: (File) -> Unit
    ) : RecyclerView.Adapter<FileAdapter.FileViewHolder>() {

        var pendingDownloadFile: File? = null

        class FileViewHolder(view: View) : RecyclerView.ViewHolder(view) {
            val fileName: TextView = view.findViewById(R.id.fileName)
            val fileInfo: TextView = view.findViewById(R.id.fileInfo)
            val shareButton: Button = view.findViewById(R.id.shareButton)
            val downloadButton: Button = view.findViewById(R.id.downloadButton)
            val deleteButton: Button = view.findViewById(R.id.deleteButton)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): FileViewHolder {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_file, parent, false)
            return FileViewHolder(view)
        }

        override fun onBindViewHolder(holder: FileViewHolder, position: Int) {
            val file = files[position]
            holder.fileName.text = file.name
            
            val sizeKB = file.length() / 1024
            val date = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US)
                .format(Date(file.lastModified()))
            holder.fileInfo.text = "$sizeKB KB • $date"

            holder.shareButton.setOnClickListener { onShare(file) }
            holder.downloadButton.setOnClickListener { onDownload(file) }
            holder.deleteButton.setOnClickListener { onDelete(file) }
        }

        override fun getItemCount() = files.size
    }
}
