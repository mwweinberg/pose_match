Inspired? by https://medium.com/tensorflow/move-mirror-an-ai-experiment-with-pose-estimation-in-the-browser-using-tensorflow-js-2f7b769f9b23


https://docs.ml5js.org/#/reference/bodypose


/archive are old versions that might be helful for troubleshooting/figuring out how you got here

## Development Plan

### Phase 3 (easy for others to add their own images and branding)

### Phase 2 (working at scale)

1. Analyze a folder with images and metadata.  Copy images with detectable human poses into a /library folder and create json or other document with associated metadata
2. Determine how to store images and metadata so they can be quickly matched and displayed
3. Implement high-speed matching to display matches in near real time
4. Improve UX to display information from metadata with the image
5. Create a standard image page to provide key details about arbitrary images based on a unique identifier
6. Add branding and styling to primary page




### Phase 1 (working prototype)

1. create page that successfully identifies point detection for 3 images and accurately draws bounding boxes around them. Page allows user to toggle through the images with keypoints and bounding box displayed to confirm that the keypoints and bounding box are correct
2. for each image, a) resize and scale around a consistent square size, b) use L2 normalization for all of the keypoints, c) save the cosine similarity score for the normalized keypoints in an external file that can be referenced later
3. create page that tracks a user through a webcam. The page displays the user and the images with the closest cosine similarity side by side. It does this by a) capturing the user image every second, performing the resizing, scaling, normalization, and cosine similarity analysis used in step 2, b) comparing the resulting cosine similarity to those of the three images from step 2, and 3) displaying the correct image