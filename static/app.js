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
  let camera, scene, renderer, avatar, mixer;
  let videoElement, videoTexture, videoMaterial;
  let clock = new THREE.Clock();
  let isThreeInitialized = false;

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
  socket.on("update_ar", function (chatbotResponse) {
    console.log("AR response received:", chatbotResponse);
    response.textContent = chatbotResponse;

    // Update speech bubble text
    updateSpeechBubble(chatbotResponse);

    // Make avatar "talk" if initialized
    if (avatar) {
      animateAvatar();
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

    // Load 3D avatar model
    const loader = new THREE.GLTFLoader();
    loader.load(
      "/models/avatar.glb",
      // onLoad callback
      function (gltf) {
        avatar = gltf.scene;

        // Scale and position the avatar
        avatar.scale.set(1, 1, 1);
        avatar.position.set(0, -1, 0);

        scene.add(avatar);

        // Check if model has animations
        if (gltf.animations && gltf.animations.length) {
          mixer = new THREE.AnimationMixer(avatar);
          const idleAnimation = gltf.animations[0];
          mixer.clipAction(idleAnimation).play();
        }

        console.log("Avatar model loaded successfully");
      },
      // onProgress callback
      function (xhr) {
        console.log((xhr.loaded / xhr.total) * 100 + "% loaded");
      },
      // onError callback
      function (error) {
        console.error("Error loading model:", error);
      }
    );

    // Handle window resize
    window.addEventListener("resize", onWindowResize);

    // Animation loop
    animate();

    isThreeInitialized = true;
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

  // Function to update speech bubble
  function updateSpeechBubble(text) {
    if (!isThreeInitialized || !avatar) return;

    speechText.textContent = text;
    speechBubble.style.display = "block";

    // Position the speech bubble above the avatar in 3D space
    const vector = new THREE.Vector3(0, 2, 0);
    vector.project(camera);

    const widthHalf = window.innerWidth / 2;
    const heightHalf = window.innerHeight / 2;

    speechBubble.style.left =
      vector.x * widthHalf + widthHalf - speechBubble.offsetWidth / 2 + "px";
    speechBubble.style.top =
      -(vector.y * heightHalf) + heightHalf - speechBubble.offsetHeight + "px";
  }

  // Function to animate avatar for "talking"
  function animateAvatar() {
    if (!avatar) return;

    // Simple head movement animation
    const duration = 1000; // 1 second
    const startTime = Date.now();

    function headAnimation() {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      if (progress < 1) {
        // Simple head rotation
        const rotationY = Math.sin(progress * Math.PI * 4) * 0.2;
        avatar.rotation.y = rotationY;

        requestAnimationFrame(headAnimation);
      } else {
        // Reset rotation when animation is complete
        avatar.rotation.y = 0;
      }
    }

    headAnimation();
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
