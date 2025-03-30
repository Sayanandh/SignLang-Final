// Common utility functions
function showLoading(button) {
    button.disabled = true;
    const spinner = button.querySelector('.fa-spinner');
    if (spinner) {
        spinner.classList.remove('hidden');
    }
}

function hideLoading(button) {
    button.disabled = false;
    const spinner = button.querySelector('.fa-spinner');
    if (spinner) {
        spinner.classList.add('hidden');
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4 fade-in';
    errorDiv.innerHTML = `
        <span class="block sm:inline">${message}</span>
        <span class="absolute top-0 bottom-0 right-0 px-4 py-3">
            <svg class="fill-current h-6 w-6 text-red-500" role="button" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                <title>Close</title>
                <path d="M14.348 14.849a1.2 1.2 0 0 1-1.697 0L10 11.819l-2.651 3.029a1.2 1.2 0 1 1-1.697-1.697l2.758-3.15-2.759-3.152a1.2 1.2 0 1 1 1.697-1.697L10 8.183l2.651-3.031a1.2 1.2 0 1 1 1.697 1.697l-2.758 3.152 2.758 3.15a1.2 1.2 0 0 1 0 1.698z"/>
            </svg>
        </span>
    `;
    
    document.querySelector('main').insertBefore(errorDiv, document.querySelector('main').firstChild);
    
    errorDiv.querySelector('svg').addEventListener('click', () => errorDiv.remove());
    setTimeout(() => errorDiv.remove(), 5000);
}

function updateConfidenceBar(confidence) {
    const confidenceBar = document.querySelector('.confidence-bar-fill');
    if (confidenceBar) {
        confidenceBar.style.width = `${confidence}%`;
    }
}

// File handling
function handleFileUpload(file, previewElement, allowedTypes) {
    if (!file) return false;
    
    const fileType = file.type.split('/')[0];
    if (!allowedTypes.includes(fileType)) {
        showError(`Invalid file type. Please upload ${allowedTypes.join(' or ')}`);
        return false;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        if (fileType === 'video') {
            previewElement.src = URL.createObjectURL(file);
        } else {
            previewElement.src = e.target.result;
        }
    };
    reader.readAsDataURL(file);
    return true;
}

// API calls
async function predictSign(formData) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        return await response.json();
    } catch (error) {
        throw new Error('Error making prediction: ' + error.message);
    }
}

// Settings
document.querySelector('.fa-cog').addEventListener('click', () => {
    // Implement settings modal
    alert('Settings functionality coming soon!');
}); 