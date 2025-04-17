document.addEventListener("DOMContentLoaded", function () {
  // DOM elements
  const connectionStatus = document.getElementById("connection-status");
  const cameraStatus = document.getElementById("camera-status");
  const currentWord = document.getElementById("current-word");
  const sentence = document.getElementById("sentence");
  const response = document.getElementById("response");
  const testArBtn = document.getElementById("test-ar-btn");
  const toggleArBtn = document.getElementById("toggle-ar-btn");
  const threeContainer = document.getElementById("three-container");
  const speechBubble = document.getElementById("speech-bubble");
  const speechText = document.getElementById("speech-text");

  // Socket.IO setup
  const socket = io("http://localhost:8000");
  console.log("Attempting to connect to Socket.IO server...");

  // Three.js variables
  let camera, scene, renderer, mixer;
  let videoElement, videoTexture, videoMaterial;
  let clock = new THREE.Clock();
  let isThreeInitialized = false;
  let currentLetterMesh = null;

  // Letter position and animation variables
  const letterPosition = new THREE.Vector3(0, 0, -2);
  const letterRotation = new THREE.Vector3(0, 0, 0);
  let letterAnimationFrame = null;

  // Connection events
  socket.on("connect", function () {
    connectionStatus.textContent = "Connected";
    connectionStatus.className = "connected";
    console.log("Connected to server");

    // Test connection
    socket.emit("check_connection");
  });

  socket.on("disconnect", function () {
    connectionStatus.textContent = "Disconnected";
    connectionStatus.className = "disconnected";
    console.log("Disconnected from server");
  });

  socket.on("connection_status", function (data) {
    console.log("Connection status:", data.status);
    if (data.status === "active" || data.status === "connected") {
      connectionStatus.textContent = "Connected";
      connectionStatus.className = "connected";
    }
  });

  // Sign language recognition updates
  socket.on("update_signs", function (data) {
    currentWord.textContent = data.current_word || "None";
    sentence.textContent = data.sentence || "None";
    console.log("Signs updated:", data);
  });

  // AR response updates
  socket.on("update_ar", function (letter) {
    console.log("AR response received:", letter);
    response.textContent = letter;

    // If 3D view is active, show the letter
    if (isThreeInitialized) {
      // Create and display 3D letter
      createLetterModel(letter);
    }
  });

  // Error handling
  socket.on("error", function (data) {
    console.error("Server error:", data.message);
    cameraStatus.textContent = "Error: " + data.message;
    cameraStatus.className = "error";
  });

  // Debug messages
  socket.on("debug", function (data) {
    console.log("Debug:", data);
  });

  // Test 3D button
  testArBtn.addEventListener("click", function () {
    console.log("Testing 3D display");
    socket.emit("test_ar");
  });

  // Toggle 3D View button
  toggleArBtn.addEventListener("click", function () {
    toggle3DView();
  });

  // Function to initialize Three.js scene
  function initThree() {
    if (isThreeInitialized) return;

    console.log("Initializing Three.js...");

    // Create scene
    scene = new THREE.Scene();

    // Create camera
    camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.z = 5;

    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    threeContainer.appendChild(renderer.domElement);

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(0, 1, 1);
    scene.add(directionalLight);

    // Add webcam video background
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then(function (stream) {
        videoElement = document.createElement("video");
        videoElement.srcObject = stream;
        videoElement.play();

        // Create video texture
        videoTexture = new THREE.VideoTexture(videoElement);

        // Create a plane for the video
        const videoGeometry = new THREE.PlaneGeometry(16, 9);
        videoGeometry.scale(1, 1, 1);
        videoMaterial = new THREE.MeshBasicMaterial({ map: videoTexture });

        const videoMesh = new THREE.Mesh(videoGeometry, videoMaterial);
        videoMesh.position.z = -10;

        // Scale video to fit screen
        const videoAspect = 16 / 9;
        const windowAspect = window.innerWidth / window.innerHeight;

        if (windowAspect > videoAspect) {
          videoMesh.scale.set(1 * (windowAspect / videoAspect), 1, 1);
        } else {
          videoMesh.scale.set(1, 1 / (windowAspect / videoAspect), 1);
        }

        scene.add(videoMesh);

        cameraStatus.textContent = "Camera active";
        cameraStatus.className = "active";
      })
      .catch(function (error) {
        console.error("Unable to access the camera:", error);
        cameraStatus.textContent = "Camera error: " + error.message;
        cameraStatus.className = "error";
      });

    // Handle window resize
    window.addEventListener("resize", onWindowResize);

    // Animation loop
    animate();

    isThreeInitialized = true;
  }

  // Create 3D letter model
  function createLetterModel(letter) {
    // Remove any existing letter model
    if (currentLetterMesh) {
      scene.remove(currentLetterMesh);
      if (letterAnimationFrame) {
        cancelAnimationFrame(letterAnimationFrame);
      }
    }

    // Create text geometry
    const fontLoader = new THREE.FontLoader();

    // Use a predefined font since we don't want to load external fonts
    fontLoader.load('https://threejs.org/examples/fonts/helvetiker_bold.typeface.json', function(font) {
      const textGeometry = new THREE.TextGeometry(letter, {
        font: font,
        size: 1,
        height: 0.2,
        curveSegments: 12,
        bevelEnabled: true,
        bevelThickness: 0.03,
        bevelSize: 0.02,
        bevelOffset: 0,
        bevelSegments: 5
      });

      // Center the text geometry
      textGeometry.computeBoundingBox();
      const centerOffset = -0.5 * (textGeometry.boundingBox.max.x - textGeometry.boundingBox.min.x);

      // Create material
      const textMaterial = new THREE.MeshPhongMaterial({
        color: 0x00aaff,
        specular: 0x111111,
        shininess: 30
      });

      // Create mesh
      currentLetterMesh = new THREE.Mesh(textGeometry, textMaterial);
      currentLetterMesh.position.copy(letterPosition);
      currentLetterMesh.position.x += centerOffset;
      currentLetterMesh.rotation.copy(letterRotation);

      scene.add(currentLetterMesh);

      // Animate letter entry
      animateLetter();
    });
  }

  // Function to animate letter entry
  function animateLetter() {
    if (!currentLetterMesh) return;

    const startScale = 0.1;
    const endScale = 1.0;
    const animationDuration = 1000; // ms
    const startTime = Date.now();

    currentLetterMesh.scale.set(startScale, startScale, startScale);

    function updateLetterAnimation() {
      const now = Date.now();
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / animationDuration, 1);

      // Ease in out
      const easeProgress = progress < 0.5
        ? 2 * progress * progress
        : 1 - Math.pow(-2 * progress + 2, 2) / 2;

      const scale = startScale + (endScale - startScale) * easeProgress;
      currentLetterMesh.scale.set(scale, scale, scale);

      // Rotate slowly
      currentLetterMesh.rotation.y = Math.sin(elapsed / 500) * 0.2;

      if (progress < 1) {
        letterAnimationFrame = requestAnimationFrame(updateLetterAnimation);
      }
    }

    letterAnimationFrame = requestAnimationFrame(updateLetterAnimation);
  }

  // Resize handler
  function onWindowResize() {
    if (!isThreeInitialized) return;

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);

    // Update video background scale if needed
    if (videoElement) {
      const videoAspect = 16 / 9;
      const windowAspect = window.innerWidth / window.innerHeight;

      if (windowAspect > videoAspect) {
        scene.children[2].scale.set(1 * (windowAspect / videoAspect), 1, 1);
      } else {
        scene.children[2].scale.set(1, 1 / (windowAspect / videoAspect), 1);
      }
    }
  }

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);

    // Update video texture if available
    if (videoTexture) videoTexture.needsUpdate = true;

    // Update mixer for animations
    if (mixer) {
      mixer.update(clock.getDelta());
    }

    renderer.render(scene, camera);
  }

  // Function to toggle 3D view
  function toggle3DView() {
    if (threeContainer.style.display === "none") {
      // Show 3D view
      threeContainer.style.display = "block";
      document.body.classList.add("three-mode");
      toggleArBtn.textContent = "Hide 3D View";

      // Initialize Three.js if not already
      if (!isThreeInitialized) {
        initThree();
      }
    } else {
      // Hide 3D view
      threeContainer.style.display = "none";
      speechBubble.style.display = "none";
      document.body.classList.remove("three-mode");
      toggleArBtn.textContent = "Show 3D View";
    }
  }
});