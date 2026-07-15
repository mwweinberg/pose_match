

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
let primaryPose = null;       // The pose we match against (largest person in frame)
let bestMatchImg = null;
let bestMatchData = null;      // Match we are switching to (image may still be loading)
let displayedMatchData = null; // Match whose image is actually on screen — the
                               // label, QR code, and announcements follow this so
                               // they never describe an image that hasn't loaded yet
let referenceList = null;      // Pre-processed reference poses (built on first match)

// QR code variables
let qrCode = null;
let currentQRObjectId = null;  // Track what QR is currently showing

// Accessibility: debounce timer for screen reader announcements
let announceTimeout = null;
const ANNOUNCE_DELAY_MS = 1500;  // Wait 1.5 seconds of stable match before announcing

// Camera status
let cameraReady = false;
let cameraError = false;

// Webcam crop parameters (set during draw, used to map skeleton keypoints)
let cropSx = 0, cropSy = 0, cropSw = 640, cropSh = 480;
let cropAreaW = 400, cropAreaH = 593;

// Explore mode: penalize recently shown images to encourage variety
// Without this, the same ~15 images dominate because ~71% of the dataset
// shares a nearly identical standing pose (cosine similarity > 0.95)
let exploreMode = true;
let recentMatches = [];
// How many recently shown images to remember and exclude from matching.
// Once an image falls off this list, it can be shown again.
// Sized close to the dataset (~1300 images) so a long session cycles
// through most of the collection before repeating.
// Recent images are excluded outright rather than score-penalized: for
// distinctive poses the dataset has only a few close matches and similarity
// falls off a cliff (0.1-0.45) right after them, so no fixed penalty can
// stop those few from winning over and over.
const EXPLORE_HISTORY_SIZE = 400;

// In explore mode, instead of always showing the single highest-scoring image,
// pick randomly among all near-tie candidates. For common poses, hundreds of
// images score within 0.02 of the top match — a difference no user can
// perceive — so sampling from that band spreads matches across the whole
// collection instead of the same few argmax winners.
const MATCH_SAMPLE_EPSILON = 0.015;  // candidates within this of the best score
const MATCH_SAMPLE_TOP_K = 40;       // cap on the candidate pool per pick

// Movement-aware timing for explore mode:
// Instead of penalizing images immediately, wait a minimum display time
// so the user can see each match before it rotates away.
let matchStartTime = 0;        // When the current match first appeared
let previousPoseVector = null;  // Previous frame's pose vector for movement detection
// Minimum time (ms) to show each image before it can be penalized away.
// When standing still, images stay longer; when moving, they rotate faster.
const EXPLORE_MIN_DISPLAY_STILL_MS = 5000;   // 5 seconds when not moving
const EXPLORE_MIN_DISPLAY_MOVING_MS = 1000;  // 1 second when actively moving
// Cosine similarity between consecutive pose vectors above this threshold
// means the person is standing still (vectors are nearly identical).
const MOVEMENT_THRESHOLD = 0.98;

// Keypoints below this confidence are ignored for matching and drawing —
// MoveNet reports essentially random positions for occluded joints.
const KEYPOINT_CONFIDENCE_MIN = 0.1;
// Require at least this many confident keypoints before matching at all.
// Fewer means the pose vector is mostly noise (person half out of frame).
const MIN_CONFIDENT_KEYPOINTS = 8;

// Clear the explore history after this long with nobody in frame, so each
// new visitor starts fresh with the collection's best matches available.
const IDLE_RESET_MS = 30000;
let lastPoseSeenTime = 0;

// Cap on decoded images kept in memory (evicted least-recently-used).
// Uncapped, a kiosk running all day grows without bound and crashes the tab.
const IMAGE_CACHE_MAX = 40;
let imageCacheOrder = [];  // filenames, most recently used last

// Maps each keypoint to its horizontal mirror (left/right swapped) in the
// MoveNet keypoint order. Used to also match against mirrored reference
// poses: a visitor raising their left arm should match artworks with either
// arm raised — this effectively doubles the pose coverage of the dataset.
const KEYPOINT_FLIP_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15];

function preload() {
  // Load the bodyPose model
  bodyPose = ml5.bodyPose();

  // Load the pre-computed pose data for all reference images
  referencePoseData = loadJSON("input_image_metadata.json");
}

function setup() {
  // Remove loading placeholder
  let loadingMsg = document.getElementById('loading-message');
  if (loadingMsg) loadingMsg.remove();

  // Double width: webcam on left, matching image on right
  let canvas = createCanvas(800, 593);
  canvas.parent('canvas-container');

  // Accessibility: add aria-label to canvas
  canvas.elt.setAttribute('aria-label', 'Pose matching experience: your webcam feed on the left detects your pose, matched artwork appears on the right. Click the artwork or scan the QR code to learn more.');

  // Get the skeleton connection information
  connections = bodyPose.getSkeleton();

  // Set up webcam with error handling
  video = createCapture(VIDEO, function() {
    // Camera access granted
    cameraReady = true;
    // Start continuous pose detection on webcam
    bodyPose.detectStart(video, gotWebcamPoses);
  });
  video.size(640, 480);
  video.hide();

  // Handle camera permission denied
  video.elt.addEventListener('error', function() {
    cameraError = true;
  });

  // Also check via getUserMedia for permission denied
  navigator.mediaDevices.getUserMedia({ video: true })
    .catch(function(err) {
      cameraError = true;
      console.log('Camera access denied:', err.name);
    });

  // Initialize QR code (empty initially)
  qrCode = new QRCode(document.getElementById("qrcode"), {
    width: 80,
    height: 80,
    colorDark: "#000000",
    colorLight: "#ffffff"
  });

}

function draw() {
  // Only update the match at the configured interval (not every frame)
  let currentTime = millis();
  if (currentTime - lastMatchTime >= MATCH_UPDATE_INTERVAL_MS) {
    findBestMatch();
    lastMatchTime = currentTime;
  }

  // Nobody in frame for a while: clear the explore history so the next
  // visitor starts fresh, with the collection's best matches available
  if (recentMatches.length > 0 && lastPoseSeenTime > 0 &&
      currentTime - lastPoseSeenTime > IDLE_RESET_MS) {
    recentMatches = [];
    previousPoseVector = null;
  }

  // Draw webcam on the left side (or error message if camera denied)
  if (cameraError) {
    // Show error message
    fill(40);
    noStroke();
    rect(0, 0, 400, 593);
    fill(255);
    textAlign(CENTER, CENTER);
    textSize(18);
    text("Camera access denied", 200, 270);
    textSize(14);
    fill(180);
    text("Please allow camera access\nto use Pose Match", 200, 320);
  } else if (video) {
    // Crop-to-fit: scale webcam to fill area, clipping overflow (like object-fit: cover)
    let vidRatio = video.width / video.height;
    let areaW = 400;
    let areaH = 593;
    let areaRatio = areaW / areaH;
    let sx, sy, sw, sh;
    if (vidRatio > areaRatio) {
      // Webcam is wider than area — crop sides
      sh = video.height;
      sw = video.height * areaRatio;
      sx = (video.width - sw) / 2;
      sy = 0;
    } else {
      // Webcam is taller than area — crop top/bottom
      sw = video.width;
      sh = video.width / areaRatio;
      sx = 0;
      sy = (video.height - sh) / 2;
    }
    // Mirror the webcam horizontally — people expect to see themselves as
    // in a mirror; an unmirrored feed reads as "it's not following me"
    push();
    translate(areaW, 0);
    scale(-1, 1);
    image(video, 0, 0, areaW, areaH, sx, sy, sw, sh);
    pop();
    // Store crop params for skeleton mapping
    cropSx = sx; cropSy = sy; cropSw = sw; cropSh = sh;
    cropAreaW = areaW; cropAreaH = areaH;
  }

  // Draw skeleton for the person being matched (mirrored to match the video)
  if (primaryPose) {
    let pose = primaryPose;
    let mapX = x => cropAreaW - (x - cropSx) * (cropAreaW / cropSw);
    let mapY = y => (y - cropSy) * (cropAreaH / cropSh);

    // Draw the skeleton connections
    for (let j = 0; j < connections.length; j++) {
      let pointAIndex = connections[j][0];
      let pointBIndex = connections[j][1];
      let pointA = pose.keypoints[pointAIndex];
      let pointB = pose.keypoints[pointBIndex];
      // Only draw a line if both points are confident enough
      if (pointA.confidence > KEYPOINT_CONFIDENCE_MIN && pointB.confidence > KEYPOINT_CONFIDENCE_MIN) {
        stroke(255, 0, 0);
        strokeWeight(2);
        line(mapX(pointA.x), mapY(pointA.y), mapX(pointB.x), mapY(pointB.y));
      }
    }

    // Draw all the tracked landmark points
    for (let j = 0; j < pose.keypoints.length; j++) {
      let keypoint = pose.keypoints[j];
      if (keypoint.confidence > KEYPOINT_CONFIDENCE_MIN) {
        fill(0, 255, 0);
        noStroke();
        circle(mapX(keypoint.x), mapY(keypoint.y), 10);
      }
    }
  }

  // Draw the best matching image on the right side (preserve aspect ratio)
  if (bestMatchImg) {
    let areaW = 400;
    let areaH = 593;
    let imgRatio = bestMatchImg.width / bestMatchImg.height;
    let areaRatio = areaW / areaH;
    let drawW, drawH;
    if (imgRatio > areaRatio) {
      // Image is wider than area — fit to width
      drawW = areaW;
      drawH = areaW / imgRatio;
    } else {
      // Image is taller than area — fit to height
      drawH = areaH;
      drawW = areaH * imgRatio;
    }
    let drawX = 400 + (areaW - drawW) / 2;
    let drawY = (areaH - drawH) / 2;
    // Black background behind image to fill letterbox area
    fill(0);
    noStroke();
    rect(400, 0, areaW, areaH);
    image(bestMatchImg, drawX, drawY, drawW, drawH);
  }

  // Draw label for the image actually on screen (not one still loading)
  if (displayedMatchData) {
    fill(255);
    noStroke();
    textAlign(CENTER, TOP);

    // Display title
    textSize(14);
    let title = displayedMatchData.metadata.title || '';
    text(title, 410, 10, 380, 40);

    // Display artist (smaller, below title)
    textSize(12);
    fill(200);  // Slightly dimmer
    let artist = displayedMatchData.metadata.artist || '';
    text(artist, 410, 55, 380, 30);
  }
}

// Callback for continuous webcam pose detection
function gotWebcamPoses(results) {
  webcamPoses = results;
  primaryPose = selectPrimaryPose(results);
  if (primaryPose) {
    lastPoseSeenTime = millis();
  }
}

// With multiple people in frame, match the one taking up the most space
// (closest to the camera) instead of whichever the model listed first —
// otherwise the match can flip between people from frame to frame.
function selectPrimaryPose(poses) {
  let best = null;
  let bestArea = -1;
  for (let pose of poses) {
    if (!pose || !pose.keypoints) {
      continue;
    }
    let box = computeBoundingBox(pose.keypoints, KEYPOINT_CONFIDENCE_MIN);
    if (box.xMin === Infinity) {
      continue;  // no confident keypoints at all
    }
    let area = (box.xMax - box.xMin) * (box.yMax - box.yMin);
    if (area > bestArea) {
      bestArea = area;
      best = pose;
    }
  }
  return best;
}

// Build the pre-processed reference list once: unwrap the loadJSON object,
// and precompute a horizontally mirrored copy of every pose vector (and its
// keypoint confidences) so each artwork can match either chirality.
function buildReferenceList() {
  let poseArray = Array.isArray(referencePoseData) ? referencePoseData : Object.values(referencePoseData);
  referenceList = [];
  for (let reference of poseArray) {
    if (!reference.l2_vector) {
      continue;
    }
    let confidences = reference.keypoint_confidences || null;
    referenceList.push({
      data: reference,
      vector: reference.l2_vector,
      confidences: confidences,
      flippedVector: flipPoseVector(reference.l2_vector),
      flippedConfidences: confidences ? KEYPOINT_FLIP_MAP.map(i => confidences[i]) : null
    });
  }
}

// Mirror a flattened pose vector horizontally: negate x and swap left/right
// keypoints. The vectors are centered on the bounding box, so negating x
// mirrors in place; L2 magnitude is unchanged.
function flipPoseVector(vector) {
  let flipped = new Array(vector.length);
  for (let k = 0; k < KEYPOINT_FLIP_MAP.length; k++) {
    let src = KEYPOINT_FLIP_MAP[k] * 2;
    flipped[k * 2] = -vector[src];
    flipped[k * 2 + 1] = vector[src + 1];
  }
  return flipped;
}

// Find the reference pose that best matches the webcam pose
function findBestMatch() {
  // Need webcam pose and reference poses to compare
  if (primaryPose === null || referencePoseData === null) {
    return;
  }

  // Process the webcam pose
  let webcamProcessed = processPose(primaryPose);
  if (webcamProcessed === null) {
    return;
  }

  if (referenceList === null) {
    buildReferenceList();
  }

  let bestSimilarity = -Infinity;
  let bestData = null;
  let bestEligibleSimilarity = -Infinity;
  let eligible = [];  // explore mode: candidates not shown recently

  // Loop through all reference poses and find the best match
  for (let i = 0; i < referenceList.length; i++) {
    let entry = referenceList[i];

    // Compare against the artwork's pose and its mirror image, over only
    // the keypoints both sides are confident about
    let similarity = Math.max(
      maskedCosineSimilarity(webcamProcessed.l2Vector, entry.vector,
                             webcamProcessed.keypointMask, entry.confidences),
      maskedCosineSimilarity(webcamProcessed.l2Vector, entry.flippedVector,
                             webcamProcessed.keypointMask, entry.flippedConfidences)
    );

    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestData = entry.data;
    }

    // In explore mode, recently shown images are ineligible — only fresh
    // images compete. This is what guarantees variety: for distinctive poses
    // the few close matches would otherwise win every rotation.
    if (exploreMode && recentMatches.indexOf(entry.data.filename) === -1) {
      eligible.push({ similarity: similarity, reference: entry.data });
      if (similarity > bestEligibleSimilarity) {
        bestEligibleSimilarity = similarity;
      }
    }
  }

  // In explore mode, pick randomly among the near-tie eligible candidates
  // rather than always taking the single best. The min-display gate below
  // still controls when the displayed image is actually allowed to rotate.
  if (exploreMode && eligible.length > 0) {
    let band = eligible.filter(c => c.similarity >= bestEligibleSimilarity - MATCH_SAMPLE_EPSILON);
    // Don't re-pick the image already on screen if there's any alternative
    if (band.length > 1 && bestMatchData) {
      let withoutCurrent = band.filter(c => c.reference.filename !== bestMatchData.filename);
      if (withoutCurrent.length > 0) {
        band = withoutCurrent;
      }
    }
    band.sort((a, b) => b.similarity - a.similarity);
    if (band.length > MATCH_SAMPLE_TOP_K) {
      band.length = MATCH_SAMPLE_TOP_K;
    }
    // band can be empty if every similarity was NaN (degenerate pose where
    // no keypoint cleared the confidence threshold) — keep the current match
    if (band.length > 0) {
      bestData = band[Math.floor(Math.random() * band.length)].reference;
    }
  }

  // If we found a match, update the display
  if (bestData !== null) {
    let matchChanged = bestData.filename !== (bestMatchData && bestMatchData.filename);

    if (matchChanged) {
      // A different image wants to win — only allow the switch if the current image
      // has been shown for the minimum display time. This prevents rapid cycling when
      // the pose naturally fluctuates between a few similar images.
      let isMoving = previousPoseVector !== null &&
        cosineSimilarity(webcamProcessed.l2Vector, previousPoseVector) < MOVEMENT_THRESHOLD;
      let minDisplay = isMoving ? EXPLORE_MIN_DISPLAY_MOVING_MS : EXPLORE_MIN_DISPLAY_STILL_MS;

      if (bestMatchData === null || millis() - matchStartTime >= minDisplay) {
        // Allowed to switch — penalize the outgoing image now that we're done with it
        if (exploreMode && bestMatchData &&
            recentMatches.indexOf(bestMatchData.filename) === -1) {
          recentMatches.push(bestMatchData.filename);
          // Cap history below the dataset size so fresh images always remain
          // eligible — a dataset smaller than EXPLORE_HISTORY_SIZE would
          // otherwise permanently exhaust and freeze explore mode
          let historyCap = Math.min(EXPLORE_HISTORY_SIZE, Math.floor(referenceList.length / 2));
          while (recentMatches.length > historyCap) {
            recentMatches.shift();
          }
        }
        matchStartTime = millis();
        bestMatchData = bestData;
      }
      // else: minimum display time not yet elapsed — keep showing the current match
    }

    // Store current pose vector for next frame's movement detection
    previousPoseVector = webcamProcessed.l2Vector;

    // Swap in the image — and only once it's actually visible, update the
    // label/QR/announcement via displayedMatchData, so they never describe
    // an image that hasn't loaded yet
    let filename = bestMatchData.filename;
    if (imageCache[filename]) {
      bestMatchImg = imageCache[filename];
      touchImageCache(filename);
      if (displayedMatchData !== bestMatchData) {
        displayedMatchData = bestMatchData;
        updateQRCode(displayedMatchData.object_id);
      }
    } else {
      loadImage("input_images/" + filename, function(img) {
        imageCache[filename] = img;
        touchImageCache(filename);
        // Only swap in if this is still the match we're switching to
        if (bestMatchData && bestMatchData.filename === filename) {
          bestMatchImg = img;
          displayedMatchData = bestMatchData;
          updateQRCode(displayedMatchData.object_id);
        }
      });
    }
  }
}

// Mark an image as recently used and evict the least-recently-used images
// beyond the cache cap. Evicted p5.Images are garbage-collected once nothing
// references them (the on-screen image stays alive via bestMatchImg).
function touchImageCache(filename) {
  let idx = imageCacheOrder.indexOf(filename);
  if (idx !== -1) {
    imageCacheOrder.splice(idx, 1);
  }
  imageCacheOrder.push(filename);
  while (imageCacheOrder.length > IMAGE_CACHE_MAX) {
    let evicted = imageCacheOrder.shift();
    delete imageCache[evicted];
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

// Process a pose: normalize keypoints and apply L2 normalization.
// Returns null when too few keypoints are confident — a vector built mostly
// from occluded joints is noise and would produce arbitrary matches.
function processPose(pose) {
  if (!pose || !pose.keypoints) {
    return null;
  }

  let keypointMask = pose.keypoints.map(kp => kp.confidence > KEYPOINT_CONFIDENCE_MIN);
  let confidentCount = keypointMask.filter(Boolean).length;
  if (confidentCount < MIN_CONFIDENT_KEYPOINTS) {
    return null;
  }

  let normalizedKeypoints = normalizeKeypoints(pose.keypoints);
  let vector = keypointsToVector(normalizedKeypoints);
  let l2Vector = l2Normalize(vector);

  return {
    normalizedKeypoints: normalizedKeypoints,
    l2Vector: l2Vector,
    keypointMask: keypointMask
  };
}

// Cosine similarity over only the keypoints both poses are confident about.
// The webcam side supplies a boolean mask; the reference side supplies its
// per-keypoint confidences when the dataset includes them (older datasets
// don't — then only the webcam mask applies). Both vectors are renormalized
// over the shared dimensions so partial poses compare fairly.
function maskedCosineSimilarity(vectorA, vectorB, keypointMaskA, refConfidences) {
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let k = 0; k < keypointMaskA.length; k++) {
    if (!keypointMaskA[k]) {
      continue;
    }
    if (refConfidences && refConfidences[k] <= KEYPOINT_CONFIDENCE_MIN) {
      continue;
    }
    let d = k * 2;
    dot += vectorA[d] * vectorB[d] + vectorA[d + 1] * vectorB[d + 1];
    magA += vectorA[d] * vectorA[d] + vectorA[d + 1] * vectorA[d + 1];
    magB += vectorB[d] * vectorB[d] + vectorB[d + 1] * vectorB[d + 1];
  }
  if (magA === 0 || magB === 0) {
    return -1;  // no shared confident keypoints — treat as a non-match
  }
  return dot / Math.sqrt(magA * magB);
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

// Update the QR code when the match changes
function updateQRCode(objectId) {
  if (qrCode && objectId !== currentQRObjectId) {
    currentQRObjectId = objectId;
    // Build the URL for the info page
    let infoUrl = window.location.href.replace(/[^/]*$/, '') + 'info.html?id=' + objectId;
    qrCode.clear();
    qrCode.makeCode(infoUrl);

    // Debounced screen reader announcement - only announce after match is stable for a period of time (avoids a barrage of announcements)
    if (announceTimeout) {
      clearTimeout(announceTimeout);
    }
    announceTimeout = setTimeout(announceMatch, ANNOUNCE_DELAY_MS);
  }
}

// Announce the current match to screen readers
function announceMatch() {
  let announcement = document.getElementById('match-announcement');
  if (announcement && displayedMatchData) {
    let title = displayedMatchData.metadata.title || 'Untitled';
    let artist = displayedMatchData.metadata.artist || 'Unknown artist';
    announcement.textContent = 'Matched artwork: ' + title + ' by ' + artist;
  }
}

// Click on matched image to open info page
function mousePressed() {
  // Only respond to left-clicks (allow right-click context menu)
  if (mouseButton !== LEFT) {
    return;
  }

  // Only respond to clicks on the right side (matched image area)
  if (mouseX > 400 && mouseX < 800 && mouseY > 0 && mouseY < 593) {
    if (displayedMatchData) {
      let infoUrl = window.location.href.replace(/[^/]*$/, '') + 'info.html?id=' + displayedMatchData.object_id;
      window.open(infoUrl, '_blank');
    }
  }
}
