const validImageTypes = [
	'image/jpeg',
	'image/png',
	'image/jpg',
	'image/bmp',
	'image/JPG',
];
let API_KEY;
fetch('/get-api-key') // Adjusted to match the route in Flask
	.then((response) => response.json())
	.then((data) => {
		API_KEY = data.api_key;
		console.log('API Key:', API_KEY); // Use the API key as needed
	})
	.catch((error) => {
		console.error('Error fetching API key:', error);
	});

function validateFile(event) {
	const file = event.target.files[0];
	const submitBtn = document.getElementById('submitbtn');
	const nameFile = document.getElementById('namefile');

	if (file && validImageTypes.includes(file.type)) {
		nameFile.textContent = 'File selected: ' + file.name;
		submitBtn.disabled = false; // Enable submit button
		submitBtn.style.display = 'inline-block';
	} else {
		nameFile.textContent =
			'Invalid file type! Only jpg, jpeg, bmp, png allowed.';
		submitBtn.disabled = true; // Disable submit button
	}
}

function compressImage(file, callback) {
	console.log('compressing...');
	const reader = new FileReader();
	reader.readAsDataURL(file);
	reader.onload = function (event) {
		const img = new Image();
		img.src = event.target.result;
		img.onload = function () {
			const canvas = document.createElement('canvas');
			const ctx = canvas.getContext('2d');

			// Set new dimensions (scale image while maintaining aspect ratio)
			const maxWidth = 1024;
			const maxHeight = 1024;
			let width = img.width;
			let height = img.height;

			if (width > height) {
				if (width > maxWidth) {
					height = Math.round(height * (maxWidth / width));
					width = maxWidth;
				}
			} else {
				if (height > maxHeight) {
					width = Math.round(width * (maxHeight / height));
					height = maxHeight;
				}
			}

			canvas.width = width;
			canvas.height = height;
			ctx.drawImage(img, 0, 0, width, height);

			// Convert canvas back to Blob and call callback
			console.log('compressed...');
			canvas.toBlob(callback, 'image/jpeg', 0.8); // Adjust quality as needed
		};
	};
}

function removeBackground(compressedBlob, callback) {
	console.log('removing background...');
	const form = new FormData();
	form.append('image_file', compressedBlob);

	// Call the background removal API
	fetch('https://clipdrop-api.co/remove-background/v1', {
		method: 'POST',
		headers: {
			'x-api-key': API_KEY,
		},
		body: form,
	})
		.then((response) => response.arrayBuffer())
		.then((buffer) => {
			const blob = new Blob([buffer], { type: 'image/png' });
			const reader = new FileReader();
			reader.onload = function (e) {
				const bgRemovedImage = e.target.result;
				callback(bgRemovedImage); // Call the callback with bg removed image
			};
			reader.readAsDataURL(blob); // Convert the buffer to Base64 for display
		})
		.catch((error) => {
			console.error('Error removing background:', error);
			alert('Background removal failed! Please try again.');
			// Hide the loader in case of error
			document.getElementById('loader').classList.add('d-none');
		});
}

function sendImageToModel(image) {
	const data = { image: image };
	console.log('Sending image data:', data); // Log the image being sent to the backend

	fetch('/predict', {
		// Modified to relative route
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify(data),
	})
		.then((response) => response.json())
		.then((result) => {
			console.log('Prediction result:', result); // Log the result
			document.getElementById(
				'predictionResult'
			).textContent = `Prediction: ${result.prediction}`;
		})
		.catch((error) => {
			console.error('Error:', error);
			alert('Failed to send image for prediction.');
			window.location.href = '/fail'; // Updated path
		});
}

function processImage() {
	const file = document.getElementById('fileup').files[0];
	const loader = document.getElementById('loader');

	if (file && validImageTypes.includes(file.type)) {
		// Show the loader when processing starts
		loader.classList.remove('d-none');

		// Compress the image
		compressImage(file, function (compressedBlob) {
			const reader = new FileReader();
			reader.readAsDataURL(compressedBlob);
			reader.onload = function (event) {
				const compressedImage = event.target.result;
				sessionStorage.setItem('compressedImage', compressedImage);

				// Remove background
				removeBackground(compressedBlob, function (bgRemovedImage) {
					sessionStorage.setItem('bgRemovedImage', bgRemovedImage);

					const readerOriginal = new FileReader();
					readerOriginal.readAsDataURL(file);
					readerOriginal.onload = function (e) {
						sessionStorage.setItem('uploadedImage', e.target.result);

						// Hide the loader after all processing is done
						loader.classList.add('d-none');

						// Redirect to result page
						window.location.href = '/result'; // Updated path
					};
				});
			};
		});
		return false; // Prevent form submission
	}
	// If invalid file type, prevent submission
	return false;
}
