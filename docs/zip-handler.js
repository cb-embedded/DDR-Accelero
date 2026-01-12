/**
 * ZIP File Handler
 * Handles extraction and parsing of sensor data from Android Sensor Logger ZIP files
 */

class ZipHandler {
    constructor() {
        this.sensorData = null;
        this.gravityData = [];
        this.gyroscopeData = [];
        this.magnetometerData = [];
        this.timestamps = [];
    }

    /**
     * Load and parse ZIP file containing sensor data
     * @param {File} file - The ZIP file
     * @returns {Promise<Object>} Parsed sensor data
     */
    async loadZip(file) {
        try {
            const zip = await JSZip.loadAsync(file);
            
            // Extract CSV files
            const gravityCSV = await this.extractCSV(zip, 'Gravity.csv');
            const gyroscopeCSV = await this.extractCSV(zip, 'Gyroscope.csv');
            const magnetometerCSV = await this.extractCSV(zip, 'Magnetometer.csv');

            // Parse CSV data
            this.gravityData = this.parseCSV(gravityCSV);
            this.gyroscopeData = this.parseCSV(gyroscopeCSV);
            this.magnetometerData = this.parseCSV(magnetometerCSV);

            // Extract timestamps (use gravity as reference)
            this.timestamps = this.gravityData.map(row => parseFloat(row[0]));

            // Convert to seconds relative to start
            const startTime = this.timestamps[0];
            this.timestamps = this.timestamps.map(t => (t - startTime) / 1000.0);

            // Build combined sensor data structure
            this.sensorData = this.combineSensorData();

            return this.sensorData;

        } catch (error) {
            console.error('Error loading ZIP file:', error);
            const errorMsg = error?.message || String(error) || 'Unknown error loading ZIP file';
            throw new Error('Failed to load ZIP file: ' + errorMsg);
        }
    }

    /**
     * Extract CSV file from ZIP
     * @param {JSZip} zip - The ZIP object
     * @param {string} filename - Name of the CSV file
     * @returns {Promise<string>} CSV content
     */
    async extractCSV(zip, filename) {
        const file = zip.file(filename);
        if (!file) {
            throw new Error(`File ${filename} not found in ZIP`);
        }
        return await file.async('string');
    }

    /**
     * Parse CSV content
     * @param {string} csvContent - CSV file content
     * @returns {Array<Array<string>>} Parsed rows
     */
    parseCSV(csvContent) {
        const lines = csvContent.trim().split('\n');
        // Skip header row
        return lines.slice(1).map(line => {
            // Handle comma-separated values
            return line.split(',').map(v => v.trim());
        });
    }

    /**
     * Combine sensor data into a unified structure
     * @returns {Object} Combined sensor data
     */
    combineSensorData() {
        const numSamples = this.timestamps.length;

        // Create arrays for each sensor channel
        const sensors = {
            time: this.timestamps,
            acc_x: new Float32Array(numSamples),
            acc_y: new Float32Array(numSamples),
            acc_z: new Float32Array(numSamples),
            gyro_x: new Float32Array(numSamples),
            gyro_y: new Float32Array(numSamples),
            gyro_z: new Float32Array(numSamples),
            mag_x: new Float32Array(numSamples),
            mag_y: new Float32Array(numSamples),
            mag_z: new Float32Array(numSamples)
        };

        // Fill gravity (accelerometer) data
        for (let i = 0; i < Math.min(numSamples, this.gravityData.length); i++) {
            const row = this.gravityData[i];
            if (row.length >= 4) {
                sensors.acc_x[i] = parseFloat(row[1]) || 0;
                sensors.acc_y[i] = parseFloat(row[2]) || 0;
                sensors.acc_z[i] = parseFloat(row[3]) || 0;
            }
        }

        // Fill gyroscope data
        for (let i = 0; i < Math.min(numSamples, this.gyroscopeData.length); i++) {
            const row = this.gyroscopeData[i];
            if (row.length >= 4) {
                sensors.gyro_x[i] = parseFloat(row[1]) || 0;
                sensors.gyro_y[i] = parseFloat(row[2]) || 0;
                sensors.gyro_z[i] = parseFloat(row[3]) || 0;
            }
        }

        // Fill magnetometer data
        for (let i = 0; i < Math.min(numSamples, this.magnetometerData.length); i++) {
            const row = this.magnetometerData[i];
            if (row.length >= 4) {
                sensors.mag_x[i] = parseFloat(row[1]) || 0;
                sensors.mag_y[i] = parseFloat(row[2]) || 0;
                sensors.mag_z[i] = parseFloat(row[3]) || 0;
            }
        }

        return sensors;
    }

    /**
     * Get sensor data as a matrix [N x 9]
     * @returns {Float32Array} Sensor data matrix
     */
    getSensorMatrix() {
        if (!this.sensorData) return null;

        const numSamples = this.timestamps.length;
        const matrix = new Float32Array(numSamples * 9);

        for (let i = 0; i < numSamples; i++) {
            const offset = i * 9;
            matrix[offset + 0] = this.sensorData.acc_x[i];
            matrix[offset + 1] = this.sensorData.acc_y[i];
            matrix[offset + 2] = this.sensorData.acc_z[i];
            matrix[offset + 3] = this.sensorData.gyro_x[i];
            matrix[offset + 4] = this.sensorData.gyro_y[i];
            matrix[offset + 5] = this.sensorData.gyro_z[i];
            matrix[offset + 6] = this.sensorData.mag_x[i];
            matrix[offset + 7] = this.sensorData.mag_y[i];
            matrix[offset + 8] = this.sensorData.mag_z[i];
        }

        return matrix;
    }

    /**
     * Get total duration of the recording
     * @returns {number} Duration in seconds
     */
    getDuration() {
        if (!this.timestamps || this.timestamps.length === 0) return 0;
        return this.timestamps[this.timestamps.length - 1];
    }

    /**
     * Get number of samples
     * @returns {number} Number of samples
     */
    getNumSamples() {
        return this.timestamps.length;
    }

    /**
     * Get sampling rate (approximate)
     * @returns {number} Sampling rate in Hz
     */
    getSamplingRate() {
        if (this.timestamps.length < 2) return 0;
        const duration = this.getDuration();
        return this.timestamps.length / duration;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ZipHandler;
}
