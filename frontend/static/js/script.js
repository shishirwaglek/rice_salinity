// Select the canvas element
const canvas = document.getElementById('star-canvas');
const ctx = canvas.getContext('2d');

// Set canvas size to match the viewport
function resizeCanvas() {
	canvas.width = window.innerWidth;
	canvas.height = window.innerHeight;
}

// Initialize canvas size
resizeCanvas();

// Update canvas size on window resize
window.addEventListener('resize', resizeCanvas);

// Star class to define properties and behaviors
class Star {
	constructor() {
		this.reset();
	}

	reset() {
		this.x = Math.random() * canvas.width;
		this.y = Math.random() * canvas.height;
		this.radius = Math.random() * 2.5 + 0.5; // Star size between 0.5 and 2 pixels
		this.alpha = Math.random(); // Initial opacity
		this.alphaChange = Math.random() * 0.02 + 0.005; // Opacity change speed
		this.color = 'rgba(135, 206, 250,'; // Light blue color (SkyBlue)
	}

	draw() {
		ctx.beginPath();
		ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2, false);
		ctx.fillStyle = this.color + this.alpha + ')';
		ctx.fill();
	}

	update() {
		this.alpha += this.alphaChange;
		// Reverse the opacity change direction if limits are reached
		if (this.alpha <= 0 || this.alpha >= 1) {
			this.alphaChange = -this.alphaChange;
		}
	}
}

// Create an array to hold all stars
const stars = [];
const starCount = 150; // Number of stars (adjust as needed)

// Initialize stars
for (let i = 0; i < starCount; i++) {
	stars.push(new Star());
}

// Animation loop to update and draw stars
function animateStars() {
	// Clear the canvas with a transparent fill to create a fading effect
	ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'; // Slightly transparent to allow trails
	ctx.fillRect(0, 0, canvas.width, canvas.height);

	// Update and draw each star
	stars.forEach((star) => {
		star.update();
		star.draw();
	});

	// Continue the animation
	requestAnimationFrame(animateStars);
}

// Start the animation
animateStars();
