class VideoUpload {
    constructor() {
        this.dropZone = document.getElementById('drop-zone');
        this.fileInput = document.getElementById('file-input');
        this.preview = document.getElementById('preview');
        this.videoPreview = document.getElementById('video-preview');
        this.analyzeBtn = document.getElementById('analyze-btn');
        this.results = document.getElementById('results');
        this.progressBar = document.getElementById('progress-bar');
        this.progressText = document.getElementById('progress-text');
        this.timelineResults = document.getElementById('timeline-results');
        this.predictionBadge = document.getElementById('prediction-badge');
        this.currentPrediction = document.getElementById('current-prediction');
        this.exportSection = document.getElementById('export-section');
        this.exportBtn = document.getElementById('export-btn');
        
        this.videoFile = null;
        this.predictions = [];
        this.isProcessing = false;
        
        this.init();
    }
    
    init() {
        // Click to upload
        this.dropZone.addEventListener('click', () => this.fileInput.click());
        
        // File input change
        this.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            this.handleFile(file);
        });
        
        // Drag and drop
        this.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropZone.parentElement.classList.replace('border-gray-300', 'border-green-500');
        });
        
        this.dropZone.addEventListener('dragleave', () => {
            this.dropZone.parentElement.classList.replace('border-green-500', 'border-gray-300');
        });
        
        this.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropZone.parentElement.classList.replace('border-green-500', 'border-gray-300');
            const file = e.dataTransfer.files[0];
            this.handleFile(file);
        });
        
        // Analyze button
        this.analyzeBtn.addEventListener('click', () => this.analyzeVideo());
        
        // Video playback
        this.videoPreview.addEventListener('timeupdate', () => this.updatePredictionDisplay());
        
        // Export button
        this.exportBtn.addEventListener('click', () => this.exportResults());
    }
    
    handleFile(file) {
        if (!file) return;
        
        if (!file.type.startsWith('video/')) {
            showError('Please upload a video file');
            return;
        }
        
        this.videoFile = file;
        this.videoPreview.src = URL.createObjectURL(file);
        this.preview.classList.remove('hidden');
        this.results.classList.add('hidden');
        this.exportSection.classList.add('hidden');
        this.predictions = [];
        this.updateProgress(0);
    }
    
    async analyzeVideo() {
        if (!this.videoFile || this.isProcessing) return;
        
        try {
            this.isProcessing = true;
            showLoading(this.analyzeBtn);
            this.results.classList.remove('hidden');
            this.predictionBadge.classList.remove('hidden');
            
            // Process video in chunks
            const duration = this.videoPreview.duration;
            const chunkSize = 1; // Process 1 second at a time
            const totalChunks = Math.ceil(duration / chunkSize);
            
            for (let i = 0; i < totalChunks; i++) {
                const timestamp = i * chunkSize;
                const prediction = await this.processVideoFrame(timestamp);
                
                this.predictions.push({
                    timestamp,
                    ...prediction
                });
                
                this.updateProgress((i + 1) / totalChunks * 100);
                this.addTimelineResult(prediction, timestamp);
            }
            
            this.exportSection.classList.remove('hidden');
            
        } catch (error) {
            showError(error.message);
        } finally {
            this.isProcessing = false;
            hideLoading(this.analyzeBtn);
        }
    }
    
    async processVideoFrame(timestamp) {
        // Capture frame from video
        const canvas = document.createElement('canvas');
        canvas.width = this.videoPreview.videoWidth;
        canvas.height = this.videoPreview.videoHeight;
        
        this.videoPreview.currentTime = timestamp;
        await new Promise(resolve => this.videoPreview.addEventListener('seeked', resolve, { once: true }));
        
        canvas.getContext('2d').drawImage(this.videoPreview, 0, 0);
        
        // Convert to blob
        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg');
        });
        
        // Send to server
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        
        const data = await predictSign(formData);
        return data;
    }
    
    updateProgress(percent) {
        const rounded = Math.round(percent);
        this.progressBar.style.width = `${rounded}%`;
        this.progressText.textContent = `${rounded}%`;
    }
    
    addTimelineResult(prediction, timestamp) {
        const timeString = this.formatTime(timestamp);
        const element = document.createElement('div');
        element.className = 'flex items-center justify-between p-4 bg-gray-50 rounded-lg';
        element.innerHTML = `
            <div class="flex items-center">
                <span class="text-gray-500 mr-4">${timeString}</span>
                <span class="font-medium">${prediction.prediction}</span>
            </div>
            <span class="text-green-600 font-medium">${Math.round(prediction.confidence)}%</span>
        `;
        
        this.timelineResults.appendChild(element);
    }
    
    updatePredictionDisplay() {
        if (!this.predictions.length) return;
        
        const currentTime = this.videoPreview.currentTime;
        const prediction = this.predictions.find(p => 
            Math.abs(p.timestamp - currentTime) < 0.5
        );
        
        if (prediction) {
            this.currentPrediction.textContent = prediction.prediction;
            this.predictionBadge.classList.remove('hidden');
        } else {
            this.predictionBadge.classList.add('hidden');
        }
    }
    
    exportResults() {
        const results = {
            filename: this.videoFile.name,
            duration: this.videoPreview.duration,
            predictions: this.predictions
        };
        
        const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `gesture-flow-results-${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
    }
    
    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VideoUpload();
}); 