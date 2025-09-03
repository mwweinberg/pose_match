/*
 * 👋 Hello! This is an ml5.js example made and shared with ❤️.
 * Learn more about the ml5.js project: https://ml5js.org/
 * ml5.js license and Code of Conduct: https://github.com/ml5js/ml5-next-gen/blob/main/LICENSE.md
 *
 * This example demonstrates drawing skeletons on poses for the MoveNet model.
 */

//import * as fs from 'fs'

let video;
let bodyPose;
let poses = [];
let connections;

function preload() {
  // Load the bodyPose model
  bodyPose = ml5.bodyPose();
  img = loadImage("images/man_400x593.png");
  kahnImg = loadImage("images/kahn_400x593.png")
  manImg = loadImage("images/man_400x593.png");
  strattonImg = loadImage("images/stratton_400x593.png");
}

function setup() {
  createCanvas(400, 593);

  // // Create the video and hide it
  // video = createCapture(VIDEO);
  // video.size(width, height);
  // video.hide();

  // Start detecting poses in the webcam video
  //bodyPose.detectStart(img, gotPoses);
  //.detect just runs it once on a still image
  bodyPose.detect(img, gotPoses);
  //get the skeleton connection information
  connections = bodyPose.getSkeleton();
  
  //for saving data
  bodyPose.detect(kahnImg, gotPosesKahn);
  bodyPose.detect(manImg, gotPosesMan);
  bodyPose.detect(strattonImg, gotPosesStratton);
}

function draw() {
  

  // Draw the webcam video
  //image(video, 0, 0, width, height);
  image(img, 0, 0);

  //draw the skeleton connections
  for (let i = 0; i < poses.length; i++) {
    let pose = poses[i];
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
  }

  // Draw all the tracked landmark points
  for (let i = 0; i < poses.length; i++) {
    let pose = poses[i];
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
  
  // draw the bounding box
  //get the points from the bounding box
  for (let i = 0; i < poses.length; i++) {
    let pose = poses[i];
    yMin = pose.box['yMin'];
    xMin = pose.box['xMin'];
    yMax = pose.box['yMax'];
    xMax = pose.box['xMax'];
    //console.log(yMin);
    // for (let j = 0; j < pose.box.length; j++) {
    //   let boxpoint = pose.box[j];
    //   console.log(boxpoint);
    // }
  }
  // draw the points on the box
  fill(0,0,255);
  noStroke();
  circle(xMin, yMin, 10);
  circle(xMax, yMin, 10);
  circle(xMin, yMax, 10);
  circle(xMax, yMax, 10);
  
  //draw the lines on the box
  stroke(0,0,255);
  line(xMin, yMin, xMax, yMin);
  line(xMin, yMin, xMin, yMax);
  line(xMax, yMin, xMax, yMax);
  line(xMax, yMax, xMin, yMax);
  

}

// Callback function for when bodyPose outputs data
function gotPoses(results) {
  // Save the output to the poses variable
  poses = results;
}

function gotPosesKahn(results) {
  posesKahn = results;
}
function gotPosesMan(results) {
  posesMan = results;
}
function gotPosesStratton(results) {
  posesStratton = results;
}

function mousePressed() {
  console.log(posesKahn);
  console.log(posesMan);
  console.log(posesStratton);
  
  // img === khanImg;
  // poses === posesKahn;
}

function keyPressed() {
  if (key === '1') {
    img = kahnImg;
    poses = posesKahn;
  }
  if (key === '2') {
    img = manImg;
    poses = posesMan;
  }
  if (key === '3') {
    img = strattonImg;
    poses = posesStratton;
  }
}

