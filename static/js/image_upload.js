class ImageUpload {
    constructor() {
        this.dropZone = document.getElementById('drop-zone');
        this.fileInput = document.getElementById('file-input');
        this.preview = document.getElementById('preview');
        this.previewImage = document.getElementById('preview-image');
        this.processedImage = document.getElementById('processed-image');
        this.predictBtn = document.getElementById('predict-btn');
        this.result = document.getElementById('result');
        this.predictionText = document.getElementById('prediction-text');
        this.confidenceText = document.getElementById('confidence-text');
        
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
        
        // Predict button
        this.predictBtn.addEventListener('click', () => this.predict());
    }
    
    handleFile(file) {
        if (!file) return;
        
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.preview.classList.remove('hidden');
            this.result.classList.add('hidden');
            
            // Reset processed image and confidence
            this.processedImage.src = '';
            this.updateConfidence(0);
        };
        reader.readAsDataURL(file);
    }
    
    async predict() {
        const file = this.fileInput.files[0];
        if (!file) {
            showError('Please select an image first');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            showLoading(this.predictBtn);
            
            const data = await predictSign(formData);
            
            // Update processed image
            this.processedImage.src = data.preprocessed_image;
            
            // Update prediction
            this.predictionText.textContent = data.prediction;
            this.updateConfidence(data.confidence);
            
            // Show results
            this.result.classList.remove('hidden');
            
        } catch (error) {
            showError(error.message);
        } finally {
            hideLoading(this.predictBtn);
        }
    }
    
    updateConfidence(confidence) {
        const roundedConfidence = Math.round(confidence);
        this.confidenceText.textContent = `${roundedConfidence}%`;
        document.querySelector('.confidence-bar-fill').style.width = `${roundedConfidence}%`;
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ImageUpload();
}); 