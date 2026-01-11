/**
 * Arrow Visualization
 * Creates SVG-based visualization of DDR arrow patterns
 */

class ArrowVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.svg = null;
        this.width = 800;
        this.height = 800;
        this.arrowSize = 35;
        this.laneWidth = 100;
        this.pixelsPerSecond = 80;
        this.arrowNames = ['Left', 'Down', 'Up', 'Right'];
        this.arrowColors = ['#FF69B4', '#00CED1', '#FFD700', '#FF4500'];
    }

    /**
     * Create visualization comparing predictions and ground truth
     * @param {Array} predictions - Predicted arrows
     * @param {Array} groundTruth - Ground truth arrows from .sm file
     * @param {number} duration - Duration to display
     */
    visualize(predictions, groundTruth, duration) {
        // Clear previous visualization
        this.container.innerHTML = '';

        // Calculate height based on duration
        this.height = Math.max(800, duration * this.pixelsPerSecond + 100);

        // Create main container
        const vizContainer = document.createElement('div');
        vizContainer.style.display = 'flex';
        vizContainer.style.gap = '40px';
        vizContainer.style.justifyContent = 'center';
        vizContainer.style.padding = '20px';

        // Create predictions column
        const predColumn = this.createColumn('ML Predictions', predictions, duration, '#667eea');
        vizContainer.appendChild(predColumn);

        // Create ground truth column
        const gtColumn = this.createColumn('Ground Truth (.sm)', groundTruth, duration, '#764ba2');
        vizContainer.appendChild(gtColumn);

        this.container.appendChild(vizContainer);
    }

    /**
     * Create a column for arrows
     * @param {string} title - Column title
     * @param {Array} arrows - Arrow events
     * @param {number} duration - Duration in seconds
     * @param {string} color - Title color
     * @returns {HTMLElement} Column element
     */
    createColumn(title, arrows, duration, color) {
        const column = document.createElement('div');
        column.style.display = 'flex';
        column.style.flexDirection = 'column';
        column.style.alignItems = 'center';

        // Title
        const titleDiv = document.createElement('div');
        titleDiv.textContent = title;
        titleDiv.style.fontSize = '1.3em';
        titleDiv.style.fontWeight = 'bold';
        titleDiv.style.color = color;
        titleDiv.style.marginBottom = '15px';
        titleDiv.style.textAlign = 'center';
        column.appendChild(titleDiv);

        // Stats
        const statsDiv = document.createElement('div');
        statsDiv.style.fontSize = '0.9em';
        statsDiv.style.color = '#666';
        statsDiv.style.marginBottom = '10px';
        statsDiv.textContent = `${arrows.length} arrows`;
        column.appendChild(statsDiv);

        // Create SVG
        const svg = this.createSVG(arrows, duration);
        column.appendChild(svg);

        return column;
    }

    /**
     * Create SVG visualization
     * @param {Array} arrows - Arrow events
     * @param {number} duration - Duration in seconds
     * @returns {SVGElement} SVG element
     */
    createSVG(arrows, duration) {
        const height = duration * this.pixelsPerSecond;
        const width = 4 * this.laneWidth;

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', width);
        svg.setAttribute('height', height);
        svg.style.border = '2px solid #ddd';
        svg.style.borderRadius = '8px';
        svg.style.background = '#f8f9fa';

        // Draw lane separators
        for (let i = 0; i <= 4; i++) {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', i * this.laneWidth);
            line.setAttribute('y1', 0);
            line.setAttribute('x2', i * this.laneWidth);
            line.setAttribute('y2', height);
            line.setAttribute('stroke', '#ddd');
            line.setAttribute('stroke-width', '1');
            svg.appendChild(line);
        }

        // Draw lane labels at bottom
        for (let i = 0; i < 4; i++) {
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', (i + 0.5) * this.laneWidth);
            text.setAttribute('y', height - 10);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('font-size', '12');
            text.setAttribute('font-weight', 'bold');
            text.setAttribute('fill', this.arrowColors[i]);
            text.textContent = this.arrowNames[i];
            svg.appendChild(text);
        }

        // Draw time markers (every 5 seconds)
        for (let t = 0; t <= duration; t += 5) {
            const y = height - (t * this.pixelsPerSecond);
            
            // Line
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', 0);
            line.setAttribute('y1', y);
            line.setAttribute('x2', width);
            line.setAttribute('y2', y);
            line.setAttribute('stroke', '#bbb');
            line.setAttribute('stroke-width', '1');
            line.setAttribute('stroke-dasharray', '5,5');
            svg.appendChild(line);

            // Label
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', width + 5);
            text.setAttribute('y', y + 4);
            text.setAttribute('font-size', '10');
            text.setAttribute('fill', '#666');
            text.textContent = `${t}s`;
            svg.appendChild(text);
        }

        // Draw arrows
        for (const event of arrows) {
            this.drawArrowEvent(svg, event, height);
        }

        return svg;
    }

    /**
     * Draw an arrow event on the SVG
     * @param {SVGElement} svg - SVG element
     * @param {Object} event - Arrow event {time, arrows}
     * @param {number} totalHeight - Total height of SVG
     */
    drawArrowEvent(svg, event, totalHeight) {
        // Convert time to Y position (bottom to top, like StepMania)
        const y = totalHeight - (event.time * this.pixelsPerSecond);

        // Draw each arrow in the event
        for (let lane = 0; lane < 4; lane++) {
            if (event.arrows[lane] === 1) {
                const x = (lane + 0.5) * this.laneWidth;
                this.drawArrow(svg, x, y, lane);
            }
        }
    }

    /**
     * Draw a single arrow
     * @param {SVGElement} svg - SVG element
     * @param {number} x - X position (center)
     * @param {number} y - Y position (center)
     * @param {number} arrowType - Arrow type (0=Left, 1=Down, 2=Up, 3=Right)
     */
    drawArrow(svg, x, y, arrowType) {
        const size = this.arrowSize;
        const color = this.arrowColors[arrowType];

        // Create arrow shape based on type
        let points = '';
        
        switch (arrowType) {
            case 0: // Left
                points = `${x - size/2},${y} ${x + size/2},${y - size/2} ${x + size/2},${y + size/2}`;
                break;
            case 1: // Down
                points = `${x},${y + size/2} ${x - size/2},${y - size/2} ${x + size/2},${y - size/2}`;
                break;
            case 2: // Up
                points = `${x},${y - size/2} ${x - size/2},${y + size/2} ${x + size/2},${y + size/2}`;
                break;
            case 3: // Right
                points = `${x + size/2},${y} ${x - size/2},${y - size/2} ${x - size/2},${y + size/2}`;
                break;
        }

        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', points);
        polygon.setAttribute('fill', color);
        polygon.setAttribute('stroke', '#333');
        polygon.setAttribute('stroke-width', '2');
        polygon.setAttribute('opacity', '0.9');
        
        // Add tooltip
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `${this.arrowNames[arrowType]} at ${event.time.toFixed(2)}s`;
        polygon.appendChild(title);

        svg.appendChild(polygon);
    }

    /**
     * Show loading state
     */
    showLoading() {
        this.container.innerHTML = '<div class="spinner"></div><p style="text-align: center; margin-top: 20px;">Generating visualization...</p>';
    }

    /**
     * Show error message
     * @param {string} message - Error message
     */
    showError(message) {
        this.container.innerHTML = `<div class="status error" style="text-align: center; padding: 40px;">${message}</div>`;
    }

    /**
     * Clear visualization
     */
    clear() {
        this.container.innerHTML = '';
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ArrowVisualizer;
}
