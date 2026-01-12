/**
 * Inference Engine
 * Performs arrow prediction using sensor data with ONNX model
 */

class InferenceEngine {
    constructor() {
        this.session = null;
        this.isModelLoaded = false;
        this.seqLength = 198;
        this.windowSize = 2.0; // 2 seconds
        this.numChannels = 9;
        this.confidenceThreshold = 0.5;
    }

    /**
     * Initialize the inference engine
     * @returns {Promise<void>}
     */
    async initialize() {
        console.log('Initializing inference engine...');
        
        try {
            // Load ONNX model
            this.session = await ort.InferenceSession.create('model.onnx');
            this.isModelLoaded = true;
            console.log('ONNX model loaded successfully');
        } catch (error) {
            console.error('Failed to load ONNX model:', error);
            const errorMsg = error?.message || String(error) || 'Unknown error';
            throw new Error(
                'Failed to load ONNX model. Please ensure:\n' +
                '1. model.onnx and model.onnx.data files exist in the docs/ directory\n' +
                '2. Files are accessible from the web server\n' +
                '3. Run "python export_model_to_onnx.py" to generate the model\n' +
                'Error details: ' + errorMsg
            );
        }
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
        
        // If no duration specified, predict for entire duration
        if (duration === null) {
            duration = sensorData.time[sensorData.time.length - 1] - startTime;
        }

        const endTime = startTime + duration;

        // Use sliding window to predict arrows
        const stride = 0.5; // Move window by 0.5 seconds each time
        
        for (let windowStart = startTime; windowStart < endTime - this.windowSize; windowStart += stride) {
            const windowEnd = windowStart + this.windowSize;
            
            // Extract window data
            const windowData = this.extractWindow(sensorData, windowStart, windowEnd);
            
            if (windowData) {
                // Run ONNX inference
                const result = await this.runInference(windowData);
                
                if (result && result.arrows.some(a => a === 1)) {
                    predictions.push({
                        time: windowStart + this.windowSize / 2 + result.offset,
                        arrows: result.arrows,
                        confidence: result.confidence
                    });
                }
            }
        }

        return predictions;
    }

    /**
     * Extract a window of sensor data
     * @param {Object} sensorData - Full sensor data
     * @param {number} windowStart - Window start time (seconds)
     * @param {number} windowEnd - Window end time (seconds)
     * @returns {Float32Array|null} Window data shaped for model input
     */
    extractWindow(sensorData, windowStart, windowEnd) {
        // Find indices for this time window
        const startIdx = sensorData.time.findIndex(t => t >= windowStart);
        const endIdx = sensorData.time.findIndex(t => t >= windowEnd);
        
        if (startIdx === -1 || endIdx === -1 || endIdx <= startIdx) {
            return null;
        }

        // Extract sensor values for this window
        const windowLength = endIdx - startIdx;
        
        // Resample to fixed sequence length
        const resampled = this.resampleWindow(sensorData, startIdx, endIdx, this.seqLength);
        
        if (!resampled) {
            return null;
        }

        // Stack all channels: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, grav_x, grav_y, grav_z]
        // Shape: [numChannels, seqLength]
        const input = new Float32Array(this.numChannels * this.seqLength);
        
        for (let ch = 0; ch < this.numChannels; ch++) {
            for (let t = 0; t < this.seqLength; t++) {
                input[ch * this.seqLength + t] = resampled[ch][t];
            }
        }
        
        return input;
    }

    /**
     * Resample window data to fixed sequence length
     * @param {Object} sensorData - Full sensor data
     * @param {number} startIdx - Start index
     * @param {number} endIdx - End index
     * @param {number} targetLength - Target sequence length
     * @returns {Array<Array<number>>|null} Resampled data [9 channels x targetLength]
     */
    resampleWindow(sensorData, startIdx, endIdx, targetLength) {
        const channels = [
            'acc_x', 'acc_y', 'acc_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'grav_x', 'grav_y', 'grav_z'
        ];

        const resampled = [];
        
        for (const channel of channels) {
            const channelData = sensorData[channel];
            if (!channelData) {
                return null;
            }

            const windowData = channelData.slice(startIdx, endIdx);
            const resampledChannel = this.linearResample(windowData, targetLength);
            resampled.push(resampledChannel);
        }
        
        return resampled;
    }

    /**
     * Resample array to target length using linear interpolation
     * @param {Array<number>} data - Input data
     * @param {number} targetLength - Target length
     * @returns {Array<number>} Resampled data
     */
    linearResample(data, targetLength) {
        if (data.length === targetLength) {
            return data;
        }

        const resampled = new Array(targetLength);
        const ratio = (data.length - 1) / (targetLength - 1);

        for (let i = 0; i < targetLength; i++) {
            const srcIndex = i * ratio;
            const srcIndexFloor = Math.floor(srcIndex);
            const srcIndexCeil = Math.min(srcIndexFloor + 1, data.length - 1);
            const fraction = srcIndex - srcIndexFloor;

            resampled[i] = data[srcIndexFloor] * (1 - fraction) + data[srcIndexCeil] * fraction;
        }

        return resampled;
    }

    /**
     * Apply sigmoid activation function
     * @param {number} x - Input value
     * @returns {number} Sigmoid output
     */
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Run ONNX inference on a window
     * @param {Float32Array} windowData - Window data [numChannels x seqLength]
     * @returns {Promise<Object>} Prediction result {arrows, offset, confidence}
     */
    async runInference(windowData) {
        try {
            // Prepare input tensor: [batch_size=1, channels=numChannels, time_steps=seqLength]
            const inputTensor = new ort.Tensor('float32', windowData, [1, this.numChannels, this.seqLength]);
            
            // Run inference
            const feeds = { input: inputTensor };
            const results = await this.session.run(feeds);
            
            // Get outputs
            const arrowsOutput = results.arrows.data; // [1, 4]
            const offsetOutput = results.offset.data; // [1, 1]
            
            // Apply sigmoid to arrows (probabilities)
            const arrows = Array.from(arrowsOutput).map(x => this.sigmoid(x));
            
            // Threshold arrows
            const arrowsBinary = arrows.map(x => x > this.confidenceThreshold ? 1 : 0);
            
            // Get offset (time adjustment)
            const offset = offsetOutput[0];
            
            // Calculate confidence (average probability of predicted arrows)
            const activeArrows = arrows.filter((_, i) => arrowsBinary[i] === 1);
            const confidence = activeArrows.length > 0
                ? activeArrows.reduce((a, b) => a + b, 0) / activeArrows.length
                : 0;
            
            return {
                arrows: arrowsBinary,
                offset: offset,
                confidence: confidence
            };
        } catch (error) {
            console.error('Inference error:', error);
            const errorMsg = error?.message || String(error) || 'Unknown inference error';
            console.error('Detailed error:', errorMsg);
            return null;
        }
    }

}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InferenceEngine;
}
