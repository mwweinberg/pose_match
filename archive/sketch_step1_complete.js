/*
 * 👋 Hello! This is an ml5.js example made and shared with ❤️.
 * Learn more about the ml5.js project: https://ml5js.org/
 * ml5.js license and Code of Conduct: https://github.com/ml5js/ml5-next-gen/blob/main/LICENSE.md
 *
 * This example demonstrates drawing skeletons on poses for the MoveNet model.
 */

let bodyPose;
let connections;

// Current displayed image and poses
let currentImg;
let currentPoses = [];

// All images and their detected poses
let kahnImg;
let manImg;
let strattonImg;
let posesKahn = [];
let posesMan = [];
let posesStratton = [];

function preload() {
  // Load the bodyPose model
  bodyPose = ml5.bodyPose();
  kahnImg = loadImage("images/kahn_400x593.png");
  manImg = loadImage("images/man_400x593.png");
  strattonImg = loadImage("images/stratton_400x593.png");
}

function setup() {
  createCanvas(400, 593);

  // Get the skeleton connection information
  connections = bodyPose.getSkeleton();

  // Detect poses for all three images
  bodyPose.detect(kahnImg, gotPosesKahn);
  bodyPose.detect(manImg, gotPosesMan);
  bodyPose.detect(strattonImg, gotPosesStratton);

  // Set initial display to man image
  currentImg = manImg;
}

function draw() {
  // Draw the current image
  image(currentImg, 0, 0);

  // Only draw poses if we have them
  if (currentPoses.length === 0) {
    return;
  }

  let pose = currentPoses[0];

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

  // Compute bounding box from keypoints (so it contains all points)
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;

  for (let j = 0; j < pose.keypoints.length; j++) {
    let keypoint = pose.keypoints[j];
    if (keypoint.confidence > 0.1) {
      xMin = min(xMin, keypoint.x);
      xMax = max(xMax, keypoint.x);
      yMin = min(yMin, keypoint.y);
      yMax = max(yMax, keypoint.y);
    }
  }

  // Draw the bounding box
  stroke(0, 0, 255);
  strokeWeight(2);
  noFill();
  rect(xMin, yMin, xMax - xMin, yMax - yMin);

  // Draw corner points
  fill(0, 0, 255);
  noStroke();
  circle(xMin, yMin, 10);
  circle(xMax, yMin, 10);
  circle(xMin, yMax, 10);
  circle(xMax, yMax, 10);
}

// Callback functions for when bodyPose outputs data
function gotPosesKahn(results) {
  posesKahn = results;
}

function gotPosesMan(results) {
  posesMan = results;
  // Set as current poses on initial load (since manImg is the default)
  if (currentPoses.length === 0) {
    currentPoses = posesMan;
  }
}

function gotPosesStratton(results) {
  posesStratton = results;
}

function mousePressed() {
  console.log("Kahn poses:", posesKahn);
  console.log("Man poses:", posesMan);
  console.log("Stratton poses:", posesStratton);
}

function keyPressed() {
  if (key === '1') {
    currentImg = kahnImg;
    currentPoses = posesKahn;
  }
  if (key === '2') {
    currentImg = manImg;
    currentPoses = posesMan;
  }
  if (key === '3') {
    currentImg = strattonImg;
    currentPoses = posesStratton;
  }
}
