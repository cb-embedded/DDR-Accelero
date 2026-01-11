/**
 * SM (StepMania) File Parser
 * Parses .sm files to extract arrow patterns and timing information
 */

class SMParser {
    constructor() {
        this.bpm = 120; // Default BPM
        this.arrows = [];
        this.difficulty = 'Medium';
        this.difficultyLevel = 5;
    }

    /**
     * Parse SM file content
     * @param {string} content - The content of the .sm file
     * @param {number} diffLevel - Difficulty level to extract
     * @param {string} diffType - Difficulty type (e.g., 'Medium', 'Easy', 'Hard')
     * @returns {Object} Parsed data containing arrows and BPM
     */
    parse(content, diffLevel = 5, diffType = 'Medium') {
        this.difficultyLevel = diffLevel;
        this.difficulty = diffType;

        // Extract BPM
        this.extractBPM(content);

        // Extract arrows for specified difficulty
        this.extractArrows(content);

        return {
            bpm: this.bpm,
            arrows: this.arrows,
            difficulty: this.difficulty,
            difficultyLevel: this.difficultyLevel
        };
    }

    /**
     * Extract BPM from SM file
     * @param {string} content - SM file content
     */
    extractBPM(content) {
        const bpmMatch = content.match(/#BPMS:([\d.=,]+);/);
        if (bpmMatch) {
            // Parse BPM string: "0.000=120.000" format
            const bpmStr = bpmMatch[1];
            const bpmParts = bpmStr.split(',')[0].split('=');
            if (bpmParts.length >= 2) {
                this.bpm = parseFloat(bpmParts[1]);
            }
        }
    }

    /**
     * Extract arrows for specified difficulty
     * @param {string} content - SM file content
     */
    extractArrows(content) {
        this.arrows = [];

        // Split by #NOTES: to get all charts
        const notesBlocks = content.split('#NOTES:').slice(1);

        for (const block of notesBlocks) {
            const lines = block.split('\n').map(l => l.trim()).filter(l => l.length > 0);
            
            if (lines.length < 6) continue;

            // Parse chart metadata
            // Format: dance-single: | description | difficulty type | difficulty level | groove radar | chart data
            const chartType = lines[0].replace(':', '');
            const description = lines[1].replace(':', '');
            const diffType = lines[2].replace(':', '').toLowerCase();
            const diffLevelStr = lines[3].replace(':', '');
            const diffLevel = parseInt(diffLevelStr);

            // Check if this is the chart we want
            if (diffType.includes(this.difficulty.toLowerCase()) && diffLevel === this.difficultyLevel) {
                // Parse the chart data (starts at line 5)
                this.parseChartData(lines.slice(5).join('\n'));
                break;
            }
        }
    }

    /**
     * Parse chart data to extract arrow timings
     * @param {string} chartData - The chart data section
     */
    parseChartData(chartData) {
        const secPerBeat = 60.0 / this.bpm;
        
        // Split into measures (separated by commas)
        const measures = chartData.split(',');
        
        let currentTime = 0.0;

        for (let measureIdx = 0; measureIdx < measures.length; measureIdx++) {
            const measure = measures[measureIdx].trim();
            
            // Stop at semicolon (end of chart)
            if (measure.includes(';')) {
                const endIdx = measure.indexOf(';');
                const finalMeasure = measure.substring(0, endIdx);
                if (finalMeasure.length > 0) {
                    this.parseMeasure(finalMeasure, currentTime, secPerBeat);
                }
                break;
            }

            currentTime = this.parseMeasure(measure, currentTime, secPerBeat);
        }
    }

    /**
     * Parse a single measure
     * @param {string} measure - Measure data
     * @param {number} startTime - Start time of measure
     * @param {number} secPerBeat - Seconds per beat
     * @returns {number} End time of measure
     */
    parseMeasure(measure, startTime, secPerBeat) {
        const rows = measure.split('\n').map(r => r.trim()).filter(r => r.length >= 4);
        const numRows = rows.length;
        
        if (numRows === 0) return startTime;

        // Time per row in this measure (4 beats per measure)
        const timePerRow = (4 * secPerBeat) / numRows;

        for (let i = 0; i < numRows; i++) {
            const row = rows[i].padEnd(4, '0');
            const time = startTime + (i * timePerRow);

            // Check if this row has any arrows
            // Arrow format: LDUR (Left, Down, Up, Right)
            // 0 = no arrow, 1/2/3/4 = arrow types
            const hasArrows = row.split('').some(c => c !== '0' && c !== 'M');

            if (hasArrows) {
                const arrowVec = [
                    row[0] !== '0' && row[0] !== 'M' ? 1 : 0, // Left
                    row[1] !== '0' && row[1] !== 'M' ? 1 : 0, // Down
                    row[2] !== '0' && row[2] !== 'M' ? 1 : 0, // Up
                    row[3] !== '0' && row[3] !== 'M' ? 1 : 0  // Right
                ];

                this.arrows.push({
                    time: time,
                    arrows: arrowVec
                });
            }
        }

        return startTime + (4 * secPerBeat);
    }

    /**
     * Extract arrows within a specific time window
     * @param {number} startTime - Start time in seconds
     * @param {number} duration - Duration in seconds
     * @returns {Array} Filtered arrows within the time window
     */
    getArrowsInWindow(startTime, duration) {
        const endTime = startTime + duration;
        return this.arrows
            .filter(a => a.time >= startTime && a.time <= endTime)
            .map(a => ({
                time: a.time - startTime, // Make relative to window start
                arrows: a.arrows
            }));
    }

    /**
     * Get total duration of the chart
     * @returns {number} Total duration in seconds
     */
    getTotalDuration() {
        if (this.arrows.length === 0) return 0;
        return this.arrows[this.arrows.length - 1].time;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SMParser;
}
