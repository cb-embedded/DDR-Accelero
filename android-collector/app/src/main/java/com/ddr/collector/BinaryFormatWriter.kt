package com.ddr.collector

import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Writer for the DDR binary sensor data format.
 * 
 * File format:
 * - Fixed 32-byte header
 * - Stream of 21-byte sensor data records
 * 
 * Header format (32 bytes):
 * | Field           | Size (bytes) | Type      | Description                    |
 * |-----------------|--------------|-----------|--------------------------------|
 * | magic_number    | 4            | uint32    | 0x44444143 ('DDAC')            |
 * | version         | 1            | uint8     | Format version (1)             |
 * | timestamp_start | 8            | uint64    | Recording start epoch (ns)     |
 * | reserved        | 19           | uint8[19] | Reserved (zeros, for alignment)|
 * 
 * Data record format (21 bytes):
 * | Field        | Size | Type    | Description                    |
 * |--------------|------|---------|--------------------------------|
 * | sensor_type  | 1    | uint8   | 1=accel, 2=gyro                |
 * | timestamp_ns | 8    | uint64  | Sample timestamp (ns)          |
 * | x            | 4    | float32 | X value                        |
 * | y            | 4    | float32 | Y value                        |
 * | z            | 4    | float32 | Z value                        |
 */
class BinaryFormatWriter(private val outputStream: OutputStream) {
    
    companion object {
        const val MAGIC_NUMBER = 0x44444143 // 'DDAC'
        const val FORMAT_VERSION: Byte = 1
        const val HEADER_SIZE = 32
        const val RECORD_SIZE = 21
        
        const val SENSOR_TYPE_ACCELEROMETER: Byte = 1
        const val SENSOR_TYPE_GYROSCOPE: Byte = 2
    }
    
    private val buffer = ByteBuffer.allocate(RECORD_SIZE).order(ByteOrder.LITTLE_ENDIAN)
    
    /**
     * Writes the file header with magic number, version, and start timestamp
     */
    fun writeHeader(timestampStartNs: Long) {
        val headerBuffer = ByteBuffer.allocate(HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN)
        
        // Magic number (4 bytes)
        headerBuffer.putInt(MAGIC_NUMBER)
        
        // Version (1 byte)
        headerBuffer.put(FORMAT_VERSION)
        
        // Start timestamp (8 bytes)
        headerBuffer.putLong(timestampStartNs)
        
        // Reserved bytes (19 bytes) - fill with zeros
        for (i in 0 until 19) {
            headerBuffer.put(0)
        }
        
        outputStream.write(headerBuffer.array())
        outputStream.flush()
    }
    
    /**
     * Writes a sensor data record
     */
    fun writeRecord(sensorType: Byte, timestampNs: Long, x: Float, y: Float, z: Float) {
        buffer.clear()
        
        buffer.put(sensorType)
        buffer.putLong(timestampNs)
        buffer.putFloat(x)
        buffer.putFloat(y)
        buffer.putFloat(z)
        
        outputStream.write(buffer.array())
    }
    
    /**
     * Flushes the output stream
     */
    fun flush() {
        outputStream.flush()
    }
    
    /**
     * Closes the output stream
     */
    fun close() {
        outputStream.close()
    }
}
