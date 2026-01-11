/**
 * Inference Engine
 * Performs arrow prediction using sensor data
 * Currently uses a simplified demo algorithm
 * TODO: Replace with ONNX model inference when model is available
 */

class InferenceEngine {
    constructor() {
        this.model = null;
        this.session = null;
        this.isModelLoaded = false;
    }

    /**
     * Initialize the inference engine
     * @returns {Promise<void>}
     */
    async initialize() {
        console.log('Initializing inference engine...');
        
        // For now, we'll use a demo algorithm
        // In the future, this would load an ONNX model:
        // this.session = await ort.InferenceSession.create('model.onnx');
        
        this.isModelLoaded = true;
        console.log('Inference engine ready (demo mode)');
    }

    /**
     * Predict arrows from sensor data
     * @param {Object} sensorData - Sensor data from ZipHandler
     * @param {number} startTime - Start time for predictions (seconds)
     * @param {number} duration - Duration to predict (seconds)
     * @returns {Array<Object>} Predicted arrows with timestamps
     */
    async predict(sensorData, startTime = 0, duration = null) {
        if (!this.isModelLoaded) {
            await this.initialize();
        }

        const predictions = [];
        const samplingRate = sensorData.time.length / sensorData.time[sensorData.time.length - 1];
        
        // If no duration specified, predict for entire duration
        if (duration === null) {
            duration = sensorData.time[sensorData.time.length - 1] - startTime;
        }

        const endTime = startTime + duration;

        // Demo algorithm: Detect peaks in accelerometer magnitude
        // This is a simplified version - real implementation would use the trained model
        const predictions_data = this.detectArrowsFromAccelerometer(
            sensorData, 
            startTime, 
            endTime
        );

        return predictions_data;
    }

    /**
     * Demo algorithm: Detect arrows from accelerometer data
     * This simulates model predictions by detecting motion peaks
     * @param {Object} sensorData - Sensor data
     * @param {number} startTime - Start time
     * @param {number} endTime - End time
     * @returns {Array<Object>} Detected arrows
     */
    detectArrowsFromAccelerometer(sensorData, startTime, endTime) {
        const predictions = [];
        
        // Calculate accelerometer magnitude
        const magnitudes = [];
        for (let i = 0; i < sensorData.time.length; i++) {
            const time = sensorData.time[i];
            if (time < startTime || time > endTime) continue;

            const mag = Math.sqrt(
                sensorData.acc_x[i] ** 2 +
                sensorData.acc_y[i] ** 2 +
                sensorData.acc_z[i] ** 2
            );
            magnitudes.push({ time, mag, index: i });
        }

        // Find peaks in magnitude (local maxima above threshold)
        const threshold = this.calculateThreshold(magnitudes.map(m => m.mag));
        
        for (let i = 5; i < magnitudes.length - 5; i++) {
            const current = magnitudes[i];
            
            // Check if this is a local maximum
            let isLocalMax = true;
            for (let j = i - 3; j <= i + 3; j++) {
                if (j !== i && magnitudes[j].mag >= current.mag) {
                    isLocalMax = false;
                    break;
                }
            }

            // If local maximum and above threshold, predict an arrow
            if (isLocalMax && current.mag > threshold) {
                // Determine arrow direction based on accelerometer axes
                const arrows = this.determineArrowDirection(
                    sensorData.acc_x[current.index],
                    sensorData.acc_y[current.index],
                    sensorData.acc_z[current.index],
                    sensorData.gyro_x[current.index],
                    sensorData.gyro_y[current.index],
                    sensorData.gyro_z[current.index]
                );

                predictions.push({
                    time: current.time - startTime, // Relative to start
                    arrows: arrows,
                    confidence: Math.min(current.mag / (threshold * 2), 1.0)
                });

                // Skip next few samples to avoid duplicate detections
                i += 5;
            }
        }

        return predictions;
    }

    /**
     * Calculate adaptive threshold for peak detection
     * @param {Array<number>} magnitudes - Magnitude values
     * @returns {number} Threshold value
     */
    calculateThreshold(magnitudes) {
        if (magnitudes.length === 0) return 0;
        
        // Calculate mean and standard deviation
        const mean = magnitudes.reduce((a, b) => a + b, 0) / magnitudes.length;
        const variance = magnitudes.reduce((a, b) => a + (b - mean) ** 2, 0) / magnitudes.length;
        const std = Math.sqrt(variance);
        
        // Threshold = mean + 1.5 * std
        return mean + 1.5 * std;
    }

    /**
     * Determine arrow direction from sensor values
     * @param {number} ax - Accelerometer X
     * @param {number} ay - Accelerometer Y
     * @param {number} az - Accelerometer Z
     * @param {number} gx - Gyroscope X
     * @param {number} gy - Gyroscope Y
     * @param {number} gz - Gyroscope Z
     * @returns {Array<number>} Arrow vector [Left, Down, Up, Right]
     */
    determineArrowDirection(ax, ay, az, gx, gy, gz) {
        // Simplified direction detection based on sensor values
        // This is a demo - real model would learn these patterns
        
        const arrows = [0, 0, 0, 0]; // [Left, Down, Up, Right]
        
        // Use accelerometer and gyroscope to estimate direction
        const absAx = Math.abs(ax);
        const absAy = Math.abs(ay);
        const absGz = Math.abs(gz);

        // Find dominant direction
        const maxAcc = Math.max(absAx, absAy);
        
        if (maxAcc < 2.0) {
            // Weak signal, random arrow for demo
            const randomDir = Math.floor(Math.random() * 4);
            arrows[randomDir] = 1;
        } else if (absAx > absAy) {
            // Horizontal movement (Left or Right)
            if (ax > 0) {
                arrows[3] = 1; // Right
            } else {
                arrows[0] = 1; // Left
            }
        } else {
            // Vertical movement (Down or Up)
            if (ay > 0) {
                arrows[1] = 1; // Down
            } else {
                arrows[2] = 1; // Up
            }
        }

        // Occasionally add double arrows (10% chance)
        if (Math.random() < 0.1 && arrows.reduce((a, b) => a + b, 0) === 1) {
            const availableDirs = arrows.map((v, i) => v === 0 ? i : -1).filter(i => i >= 0);
            if (availableDirs.length > 0) {
                const randomSecond = availableDirs[Math.floor(Math.random() * availableDirs.length)];
                arrows[randomSecond] = 1;
            }
        }

        return arrows;
    }

    /**
     * Run inference with sliding window (for real-time simulation)
     * @param {Object} sensorData - Sensor data
     * @param {number} windowSize - Size of window in seconds
     * @param {number} stride - Stride between windows in seconds
     * @returns {Array<Object>} Predictions with offsets
     */
    async predictWithSlidingWindow(sensorData, windowSize = 2.0, stride = 0.1) {
        // This would be used with the actual ONNX model
        // For now, delegate to the demo algorithm
        return await this.predict(sensorData);
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InferenceEngine;
}
