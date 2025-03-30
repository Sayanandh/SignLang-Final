class LiveCamera {
    constructor() {
        this.video = document.getElementById('camera-feed');
        this.startButton = document.getElementById('start-camera');
        this.predictionBadge = document.getElementById('prediction-badge');
        this.livePrediction = document.getElementById('live-prediction');
        this.statusBadge = document.getElementById('status-badge');
        this.confidenceText = document.getElementById('confidence-text');
        this.predictionsList = document.getElementById('predictions-list');
        
        this.stream = null;
        this.isProcessing = false;
        this.predictionInterval = null;
        this.lastPrediction = null;
        
        this.init();
    }
    
    init() {
        this.startButton.addEventListener('click', () => this.toggleCamera());
        
        // Check if browser supports getUserMedia
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showError('Your browser does not support camera access');
            this.startButton.disabled = true;
            return;
        }
    }
    
    async toggleCamera() {
        if (this.stream) {
            this.stopCamera();
        } else {
            await this.startCamera();
        }
    }
    
    async startCamera() {
        try {
            // Request camera access with specific constraints
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user',
                    frameRate: { ideal: 30 }
                }
            });
            
            // Set video source and play
            this.video.srcObject = this.stream;
            await this.video.play();
            
            // Update UI
            this.startButton.innerHTML = '<i class="fas fa-stop mr-2"></i>Stop Camera';
            this.startButton.classList.replace('bg-green-500', 'bg-red-500');
            this.startButton.classList.replace('hover:bg-green-600', 'hover:bg-red-600');
            
            this.updateStatus('Active', 'bg-green-100', 'text-green-800');
            this.predictionBadge.classList.remove('hidden');
            
            // Start prediction loop
            this.startPrediction();
            
        } catch (error) {
            if (error.name === 'NotAllowedError') {
                showError('Camera access denied. Please allow camera access and try again.');
            } else {
                showError('Error accessing camera: ' + error.message);
            }
            console.error('Camera error:', error);
        }
    }
    
    stopCamera() {
        if (this.stream) {
            // Stop all tracks
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            this.video.srcObject = null;
            
            // Update UI
            this.startButton.innerHTML = '<i class="fas fa-video mr-2"></i>Start Camera';
            this.startButton.classList.replace('bg-red-500', 'bg-green-500');
            this.startButton.classList.replace('hover:bg-red-600', 'hover:bg-green-600');
            
            this.updateStatus('Not Started', 'bg-gray-100', 'text-gray-800');
            this.predictionBadge.classList.add('hidden');
            
            // Stop prediction loop
            if (this.predictionInterval) {
                clearInterval(this.predictionInterval);
                this.predictionInterval = null;
            }
            
            // Clear predictions
            this.predictionsList.innerHTML = '';
            this.updateConfidence(0);
        }
    }
    
    startPrediction() {
        let retryCount = 0;
        const maxRetries = 3;
        
        // Capture frame and send to server every 500ms
        this.predictionInterval = setInterval(async () => {
            if (this.isProcessing || !this.stream) return;
            
            try {
                this.isProcessing = true;
                const prediction = await this.getPrediction();
                
                // Reset retry count on successful prediction
                retryCount = 0;
                this.updateStatus('Active', 'bg-green-100', 'text-green-800');
                
                // Only update if prediction changed
                if (!this.lastPrediction || 
                    prediction.prediction !== this.lastPrediction.prediction ||
                    Math.abs(prediction.confidence - this.lastPrediction.confidence) > 5) {
                    this.updatePrediction(prediction);
                    this.lastPrediction = prediction;
                }
            } catch (error) {
                console.error('Prediction error:', error);
                retryCount++;
                
                if (retryCount >= maxRetries) {
                    this.updateStatus('Connection Error', 'bg-red-100', 'text-red-800');
                    showError('Unable to connect to the server. Please check your connection and try again.');
                    this.stopCamera();
                } else {
                    this.updateStatus('Retrying...', 'bg-yellow-100', 'text-yellow-800');
                }
            } finally {
                this.isProcessing = false;
            }
        }, 500);
    }
    
    async getPrediction() {
        // Capture current frame
        const canvas = document.createElement('canvas');
        canvas.width = this.video.videoWidth;
        canvas.height = this.video.videoHeight;
        canvas.getContext('2d').drawImage(this.video, 0, 0);
        
        // Convert to blob
        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', 0.8);
        });
        
        // Send to server
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }
        
        const data = await response.json();
        if (!data || typeof data.prediction === 'undefined') {
            throw new Error('Invalid response format from server');
        }
        
        return await response.json();
    }
    
    updatePrediction(data) {
        // Update live prediction badge
        this.livePrediction.textContent = data.prediction;
        
        // Update confidence
        const confidence = Math.round(data.confidence);
        this.confidenceText.textContent = `${confidence}%`;
        document.querySelector('.confidence-bar-fill').style.width = `${confidence}%`;
        
        // Add to recent predictions if not "No hand detected"
        if (data.prediction !== 'No hand detected') {
            const predictionElement = document.createElement('div');
            predictionElement.className = 'flex justify-between items-center text-sm fade-in';
            predictionElement.innerHTML = `
                <span class="text-gray-700">${data.prediction}</span>
                <span class="text-green-600 font-medium">${confidence}%</span>
            `;
            
            this.predictionsList.insertBefore(predictionElement, this.predictionsList.firstChild);
            
            // Keep only last 5 predictions
            while (this.predictionsList.children.length > 5) {
                this.predictionsList.removeChild(this.predictionsList.lastChild);
            }
        }
    }
    
    updateStatus(text, bgColor, textColor) {
        this.statusBadge.textContent = text;
        this.statusBadge.className = `px-3 py-1 rounded-full text-sm font-medium ${bgColor} ${textColor}`;
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new LiveCamera();
});