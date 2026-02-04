

let bodyPose;
let connections;

// How often to update the matching image (in milliseconds)
// Lower = more responsive but more CPU usage
// Higher = less CPU but slower updates
// Recommended: 250 (4x/sec) to 1000 (1x/sec)
let MATCH_UPDATE_INTERVAL_MS = 250;
let lastMatchTime = 0;

// Reference pose data loaded from JSON (array of objects with l2_vector and metadata)
let referencePoseData = null;

// Cache for loaded images (keyed by filename)
let imageCache = {};

// Webcam and pose matching variables
let video;
let webcamPoses = [];
let bestMatchImg = null;
let bestMatchData = null;  // Full metadata for the current best match

function preload() {
  // Load the bodyPose model
  bodyPose = ml5.bodyPose();

  // Load the pre-computed pose data for all reference images
  referencePoseData = loadJSON("person_images_metadata.json");
}

function setup() {
  // Double width: webcam on left, matching image on right
  createCanvas(800, 593);

  // Get the skeleton connection information
  connections = bodyPose.getSkeleton();

  // Set up webcam
  video = createCapture(VIDEO);
  video.size(400, 593);
  video.hide();

  // Start continuous pose detection on webcam
  bodyPose.detectStart(video, gotWebcamPoses);
}

function draw() {
  // Only update the match at the configured interval (not every frame)
  let currentTime = millis();
  if (currentTime - lastMatchTime >= MATCH_UPDATE_INTERVAL_MS) {
    findBestMatch();
    lastMatchTime = currentTime;
  }

  // Draw webcam on the left side
  if (video) {
    image(video, 0, 0, 400, 593);
  }

  // Draw skeleton on webcam feed if poses detected
  if (webcamPoses.length > 0) {
    let pose = webcamPoses[0];

    // Draw the skeleton connections
    for (let j = 0; j < connections.length; j++) {
      let pointAIndex = connections[j][0];
      let pointBIndex = connections[j][1];
      let pointA = pose.keypoints[pointAIndex];
      let pointB = pose.keypoints[pointBIndex];
      // Only draw a line if both points are confident enough
      if (pointA.confidence > 0.1 && pointB.confidence > 0.1) {
        stroke(255, 0, 0);
        strokeWeight(2);
        line(pointA.x, pointA.y, pointB.x, pointB.y);
      }
    }

    // Draw all the tracked landmark points
    for (let j = 0; j < pose.keypoints.length; j++) {
      let keypoint = pose.keypoints[j];
      // Only draw a circle if the keypoint's confidence is bigger than 0.1
      if (keypoint.confidence > 0.1) {
        fill(0, 255, 0);
        noStroke();
        circle(keypoint.x, keypoint.y, 10);
      }
    }
  }

  // Draw the best matching image on the right side
  if (bestMatchImg) {
    image(bestMatchImg, 400, 0, 400, 593);
  }

  // Draw label for the matching image (show title from metadata)
  if (bestMatchData) {
    fill(255);
    noStroke();
    textSize(14);
    textAlign(CENTER, TOP);

    // Display title with text wrapping (width=380, height=50)
    let title = bestMatchData.metadata.title;
    text(title, 410, 10, 380, 50);
  }
}

// Callback for continuous webcam pose detection
function gotWebcamPoses(results) {
  webcamPoses = results;
}

// Find the reference pose that best matches the webcam pose
function findBestMatch() {
  // Need webcam pose and reference poses to compare
  if (webcamPoses.length === 0 || referencePoseData === null) {
    return;
  }

  // Process the webcam pose
  let webcamProcessed = processPose(webcamPoses[0]);
  if (webcamProcessed === null) {
    return;
  }

  let bestSimilarity = -Infinity;
  let bestData = null;

  // Convert to array if needed (p5.js loadJSON returns object with numeric keys for arrays)
  let poseArray = Array.isArray(referencePoseData) ? referencePoseData : Object.values(referencePoseData);

  // Loop through all reference poses and find the best match
  for (let i = 0; i < poseArray.length; i++) {
    let reference = poseArray[i];

    // Skip if no l2_vector
    if (!reference.l2_vector) {
      continue;
    }

    let similarity = cosineSimilarity(webcamProcessed.l2Vector, reference.l2_vector);

    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestData = reference;
    }
  }

  // If we found a match, update the display
  if (bestData !== null) {
    bestMatchData = bestData;

    // Check if image is already cached
    let filename = bestData.filename;
    if (imageCache[filename]) {
      // Use cached image
      bestMatchImg = imageCache[filename];
    } else {
      // Load the image and cache it
      loadImage("person_images/" + filename, function(img) {
        imageCache[filename] = img;
        // Only update bestMatchImg if this is still the best match
        if (bestMatchData && bestMatchData.filename === filename) {
          bestMatchImg = img;
        }
      });
    }
  }
}

// Compute bounding box from keypoints with confidence threshold
function computeBoundingBox(keypoints, confidenceThreshold = 0.1) {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;

  for (let keypoint of keypoints) {
    if (keypoint.confidence > confidenceThreshold) {
      xMin = Math.min(xMin, keypoint.x);
      xMax = Math.max(xMax, keypoint.x);
      yMin = Math.min(yMin, keypoint.y);
      yMax = Math.max(yMax, keypoint.y);
    }
  }

  return { xMin, xMax, yMin, yMax };
}

// Normalize keypoints: center around bounding box and scale to unit square
function normalizeKeypoints(keypoints, confidenceThreshold = 0.1) {
  let box = computeBoundingBox(keypoints, confidenceThreshold);

  // Compute center and size of bounding box
  let centerX = (box.xMin + box.xMax) / 2;
  let centerY = (box.yMin + box.yMax) / 2;
  let boxWidth = box.xMax - box.xMin;
  let boxHeight = box.yMax - box.yMin;

  // Use the larger dimension to maintain aspect ratio
  let scale = Math.max(boxWidth, boxHeight);

  // Avoid division by zero
  if (scale === 0) {
    scale = 1;
  }

  // Normalize each keypoint: center and scale to [-0.5, 0.5] range
  let normalized = [];
  for (let keypoint of keypoints) {
    normalized.push({
      name: keypoint.name,
      x: (keypoint.x - centerX) / scale,
      y: (keypoint.y - centerY) / scale,
      confidence: keypoint.confidence
    });
  }

  return normalized;
}

// Flatten keypoints to a vector [x1, y1, x2, y2, ...] for L2 normalization
function keypointsToVector(keypoints) {
  let vector = [];
  for (let keypoint of keypoints) {
    vector.push(keypoint.x);
    vector.push(keypoint.y);
  }
  return vector;
}

// L2 normalize a vector (divide by magnitude to create unit vector)
function l2Normalize(vector) {
  let sumOfSquares = 0;
  for (let val of vector) {
    sumOfSquares += val * val;
  }
  let magnitude = Math.sqrt(sumOfSquares);

  // Avoid division by zero
  if (magnitude === 0) {
    return vector;
  }

  return vector.map(val => val / magnitude);
}

// Process a pose: normalize keypoints and apply L2 normalization
function processPose(pose) {
  if (!pose || !pose.keypoints) {
    return null;
  }

  let normalizedKeypoints = normalizeKeypoints(pose.keypoints);
  let vector = keypointsToVector(normalizedKeypoints);
  let l2Vector = l2Normalize(vector);

  return {
    normalizedKeypoints: normalizedKeypoints,
    l2Vector: l2Vector
  };
}

// Compute cosine similarity between two L2-normalized vectors
// Returns a value from -1 to 1 (1 = identical, 0 = unrelated)
function cosineSimilarity(vectorA, vectorB) {
  if (vectorA.length !== vectorB.length) {
    console.error("Vectors must have the same length");
    return 0;
  }

  let dotProduct = 0;
  for (let i = 0; i < vectorA.length; i++) {
    dotProduct += vectorA[i] * vectorB[i];
  }

  return dotProduct;
}

// Debug: log current match data when clicking
function mousePressed() {
  if (bestMatchData) {
    console.log("Current match:", bestMatchData);
  }
}
