Inspired? by https://medium.com/tensorflow/move-mirror-an-ai-experiment-with-pose-estimation-in-the-browser-using-tensorflow-js-2f7b769f9b23


https://docs.ml5js.org/#/reference/bodypose


## /archive are old versions that might be helful for troubleshooting/figuring out how you got here

## /utilities contains utilities to prep the dataset (images and metadata)
### image_cleaner is the utility to take a bunch of arbitrary images, determine if they contain people, and save metadata including pose information
It assumes:
* Images in a folder called `original_met_images`
* that the filenames for those images is `objectID.jpg`
* that there is metadata in a file called `cleaned_met_data.json` that includes the fields:
** "Object_ID"
** "title"
** "artist"
** "date"
** "department"
** "medium"
** "link"
** "repository"

It outputs:
* Images that probably have people in `/person_images`
* metadata for those images into `person_images_metadata.json`

## Development Plan

### Phase 4 (easy for others to add their own images and branding)

### Phase 3 (online testing)

1. Host somewhere that makes the site available to others

### Phase 2 (working at scale)

1. Analyze a folder with images and metadata.  Copy images with detectable human poses into a /library folder and create json or other document with associated metadata
2. Determine how to store images and metadata so they can be quickly matched and displayed
3. Implement high-speed matching to display matches in near real time
4. Improve UX to display information from metadata with the image
5. Create a standard image page to provide key details about arbitrary images based on a unique identifier
6. Add branding and styling to primary page
7. Handle external webcams (choose webcam in browser?)




### Phase 1 (working prototype)

1. create page that successfully identifies point detection for 3 images and accurately draws bounding boxes around them. Page allows user to toggle through the images with keypoints and bounding box displayed to confirm that the keypoints and bounding box are correct
2. for each image, a) resize and scale around a consistent square size, b) use L2 normalization for all of the keypoints, c) save the cosine similarity score for the normalized keypoints in an external file that can be referenced later
3. create page that tracks a user through a webcam. The page displays the user and the images with the closest cosine similarity side by side. It does this by a) capturing the user image every second, performing the resizing, scaling, normalization, and cosine similarity analysis used in step 2, b) comparing the resulting cosine similarity to those of the three images from step 2, and 3) displaying the correct image