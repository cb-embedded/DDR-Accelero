/**
 * Main Application Logic
 * Coordinates file uploads, inference, and visualization
 */

// Global state
const app = {
    zipHandler: new ZipHandler(),
    smParser: new SMParser(),
    inference: new InferenceEngine(),
    visualizer: new ArrowVisualizer('visualization'),
    
    // Data
    sensorData: null,
    smData: null,
    predictions: null,
    groundTruth: null,
    
    // UI elements
    zipFile: null,
    smFile: null,
    difficulty: 5
};

// Global error handler to catch all errors
window.addEventListener('error', (event) => {
    console.error('Uncaught error:', event.error);
    displayGlobalError('Unexpected error: ' + (event.error?.message || event.error || 'Unknown error'));
    event.preventDefault();
});

// Global unhandled promise rejection handler
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    displayGlobalError('Unexpected error: ' + (event.reason?.message || event.reason || 'Unknown error'));
    event.preventDefault();
});

/**
 * Display a global error message
 */
function displayGlobalError(message) {
    // Try to display in progress div if available
    const progressDiv = document.getElementById('progress');
    if (progressDiv) {
        progressDiv.textContent = '✗ ' + message;
        progressDiv.className = 'status error';
        progressDiv.style.display = 'block';
    } else {
        // Fallback: create alert-style message
        alert('Error: ' + message);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DDR Accelero Web App initialized');
    
    // Get UI elements
    const zipFileInput = document.getElementById('zipFile');
    const smFileInput = document.getElementById('smFile');
    const difficultySelect = document.getElementById('difficulty');
    const predictButton = document.getElementById('predictButton');
    
    // Set up event listeners
    zipFileInput.addEventListener('change', handleZipUpload);
    smFileInput.addEventListener('change', handleSmUpload);
    difficultySelect.addEventListener('change', (e) => {
        app.difficulty = parseInt(e.target.value);
        updatePredictButton();
    });
    predictButton.addEventListener('click', runInference);
    
    // Initialize inference engine
    app.inference.initialize().catch(err => {
        console.error('Failed to initialize inference:', err);
        const errorMsg = err?.message || String(err) || 'Unknown error during initialization';
        displayGlobalError('Failed to initialize inference: ' + errorMsg);
    });
});

/**
 * Handle ZIP file upload
 */
async function handleZipUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const statusDiv = document.getElementById('zipStatus');
    statusDiv.textContent = 'Loading ZIP file...';
    statusDiv.className = 'status info';
    
    try {
        app.sensorData = await app.zipHandler.loadZip(file);
        
        const duration = app.zipHandler.getDuration();
        const numSamples = app.zipHandler.getNumSamples();
        const samplingRate = app.zipHandler.getSamplingRate();
        
        statusDiv.innerHTML = `
            ✓ Loaded successfully<br>
            Duration: ${duration.toFixed(1)}s<br>
            Samples: ${numSamples}<br>
            Rate: ~${samplingRate.toFixed(0)} Hz
        `;
        statusDiv.className = 'status success';
        
        updatePredictButton();
        
    } catch (error) {
        console.error('Error loading ZIP:', error);
        const errorMsg = error?.message || String(error) || 'Unknown error loading ZIP file';
        statusDiv.textContent = '✗ Error: ' + errorMsg;
        statusDiv.className = 'status error';
        app.sensorData = null;
    }
}

/**
 * Handle SM file upload
 */
async function handleSmUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const statusDiv = document.getElementById('smStatus');
    statusDiv.textContent = 'Loading SM file...';
    statusDiv.className = 'status info';
    
    try {
        const content = await file.text();
        app.smData = app.smParser.parse(content, app.difficulty, 'Medium');
        
        const numArrows = app.smData.arrows.length;
        const duration = app.smParser.getTotalDuration();
        
        statusDiv.innerHTML = `
            ✓ Loaded successfully<br>
            BPM: ${app.smData.bpm}<br>
            Arrows: ${numArrows}<br>
            Duration: ${duration.toFixed(1)}s
        `;
        statusDiv.className = 'status success';
        
        updatePredictButton();
        
    } catch (error) {
        console.error('Error loading SM file:', error);
        const errorMsg = error?.message || String(error) || 'Unknown error loading SM file';
        statusDiv.textContent = '✗ Error: ' + errorMsg;
        statusDiv.className = 'status error';
        app.smData = null;
    }
}

/**
 * Update predict button state
 */
function updatePredictButton() {
    const button = document.getElementById('predictButton');
    // Allow prediction with just ZIP file (SM file is now optional)
    const canPredict = app.sensorData !== null;
    button.disabled = !canPredict;
}

/**
 * Run inference and display results
 */
async function runInference() {
    const progressDiv = document.getElementById('progress');
    const resultsSection = document.getElementById('results');
    
    try {
        // Show progress
        progressDiv.textContent = 'Running inference...';
        progressDiv.className = 'status info';
        resultsSection.style.display = 'none';
        app.visualizer.showLoading();
        
        // Run predictions
        console.log('Starting inference...');
        const startTime = performance.now();
        
        app.predictions = await app.inference.predict(app.sensorData);
        
        const inferenceTime = performance.now() - startTime;
        console.log(`Inference completed in ${inferenceTime.toFixed(0)}ms`);
        
        // Get ground truth arrows if SM file is available
        let duration = app.zipHandler.getDuration();
        if (app.smData) {
            duration = Math.min(
                app.zipHandler.getDuration(),
                app.smParser.getTotalDuration()
            );
            app.groundTruth = app.smParser.getArrowsInWindow(0, duration);
        } else {
            app.groundTruth = null;
        }
        
        // Display results
        displayResults(duration, inferenceTime);
        
        progressDiv.textContent = `✓ Inference complete (${inferenceTime.toFixed(0)}ms)`;
        progressDiv.className = 'status success';
        
    } catch (error) {
        console.error('Error during inference:', error);
        const errorMsg = error?.message || String(error) || 'Unknown error during inference';
        progressDiv.textContent = '✗ Error: ' + errorMsg;
        progressDiv.className = 'status error';
        app.visualizer.showError('Failed to run inference: ' + errorMsg);
    }
}

/**
 * Display inference results
 */
function displayResults(duration, inferenceTime) {
    const resultsSection = document.getElementById('results');
    const statsDiv = document.getElementById('stats');
    
    // Calculate statistics
    const numPredictions = app.predictions.length;
    
    let statsHTML = '';
    
    if (app.groundTruth && app.groundTruth.length > 0) {
        // With ground truth - show comparison metrics
        const numGroundTruth = app.groundTruth.length;
        const detectionRate = numGroundTruth > 0 
            ? (numPredictions / numGroundTruth * 100).toFixed(1)
            : 0;
        
        // Calculate accuracy (approximate - matches within 0.5s window)
        let matchedPredictions = 0;
        for (const pred of app.predictions) {
            const hasMatch = app.groundTruth.some(gt => 
                Math.abs(gt.time - pred.time) < 0.5 &&
                arraysEqual(gt.arrows, pred.arrows)
            );
            if (hasMatch) matchedPredictions++;
        }
        const accuracy = numPredictions > 0
            ? (matchedPredictions / numPredictions * 100).toFixed(1)
            : 0;
        
        statsHTML = `
            <div class="stat-card">
                <h4>Duration</h4>
                <div class="value">${duration.toFixed(1)}s</div>
            </div>
            <div class="stat-card">
                <h4>Ground Truth</h4>
                <div class="value">${numGroundTruth}</div>
            </div>
            <div class="stat-card">
                <h4>Predictions</h4>
                <div class="value">${numPredictions}</div>
            </div>
            <div class="stat-card">
                <h4>Detection Rate</h4>
                <div class="value">${detectionRate}%</div>
            </div>
            <div class="stat-card">
                <h4>Approximate Accuracy</h4>
                <div class="value">${accuracy}%</div>
            </div>
            <div class="stat-card">
                <h4>Inference Time</h4>
                <div class="value">${inferenceTime.toFixed(0)}ms</div>
            </div>
        `;
    } else {
        // Without ground truth - show only prediction metrics
        statsHTML = `
            <div class="stat-card">
                <h4>Duration</h4>
                <div class="value">${duration.toFixed(1)}s</div>
            </div>
            <div class="stat-card">
                <h4>Predictions</h4>
                <div class="value">${numPredictions}</div>
            </div>
            <div class="stat-card">
                <h4>Inference Time</h4>
                <div class="value">${inferenceTime.toFixed(0)}ms</div>
            </div>
            <div class="info-message" style="grid-column: 1 / -1; padding: 15px; background: #fff3cd; border-radius: 8px; color: #856404; text-align: center;">
                <strong>Note:</strong> Upload a .sm file to see accuracy metrics and ground truth comparison
            </div>
        `;
    }
    
    statsDiv.innerHTML = statsHTML;
    
    // Create visualization
    app.visualizer.visualize(app.predictions, app.groundTruth || [], duration);
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Helper: Check if two arrays are equal
 */
function arraysEqual(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

/**
 * Helper: Calculate timing error between predictions and ground truth
 */
function calculateTimingError(predictions, groundTruth) {
    const errors = [];
    
    for (const pred of predictions) {
        // Find closest ground truth arrow with same pattern
        let minError = Infinity;
        
        for (const gt of groundTruth) {
            if (arraysEqual(pred.arrows, gt.arrows)) {
                const error = Math.abs(pred.time - gt.time);
                if (error < minError) {
                    minError = error;
                }
            }
        }
        
        if (minError < 1.0) { // Only count if within 1 second
            errors.push(minError);
        }
    }
    
    return errors;
}

// Export for debugging
if (typeof window !== 'undefined') {
    window.app = app;
}
