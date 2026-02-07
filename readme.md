Inspired? by https://medium.com/tensorflow/move-mirror-an-ai-experiment-with-pose-estimation-in-the-browser-using-tensorflow-js-2f7b769f9b23


https://docs.ml5js.org/#/reference/bodypose


## /archive are old versions that might be helpful for troubleshooting/figuring out how you got here

## Pages and Scripts

*index.html and sketch.js* are the main parts of the page
*info.html* is a template page for individual images. It expects to load with info.html?id=[object_id]
*about.html* is an about page
*analytics.js* allows you to add analytics. These are not required. 

## /utilities contains utilities to prep the dataset (images and metadata)

### image_cleaner 
is the utility to take a bunch of arbitrary images, determine if they contain people, and save metadata including pose information
It assumes:
* Images in a folder called `original_met_images`
* that the filenames for those images is `objectID.jpg`
* that there is metadata in a file called `cleaned_met_data.json` that includes the fields:
    * "Object_ID"
    * "title"
    * "artist"
    * "date"
    * "department"
    * "medium"
    * "link"
    * "repository"

It outputs:
* Images that probably have people in `/person_images`
* metadata for those images into `person_images_metadata.json`

### met_random_downloader 
downloads a random collection of images of people from the met and bundles them with the relevant metadata. This is just a script to build the original testing dataset.  Run it with `caffeinate -i python met_random_downloader.py` to avoid crashes because the compute goes to sleep in the middle

*Run met_random_downloader.py* (from the met_random_downloader/ folder)

Outputs: output_images/*.jpg and cleaned_met_data.json

*Move files to image_cleaner/:*

* output_images/*.jpg → image_cleaner/input_images/
* cleaned_met_data.json → image_cleaner/input_metadata.json **(RENAME)**

Run person_or_not.py (from the image_cleaner/ folder)

Outputs: processed_images/*.jpg and image_metadata.json

*Move files to main project root (DON'T FORGET TO RENAME THE METADATA FILE):*

* processed_images/*.jpg → pose_match/input_images/
* image_metadata.json → pose_match/input_images_metadata.json **(RENAME)**
Open index.html and it should work

## Development Plan

### Phase 4 (easy for others to add their own images and branding)

### Phase 3 (online testing)

0. Add a glam-e lab/EC footer
1. Host somewhere that makes the site available to others

### Phase 2 (working at scale)

1. Analyze a folder with images and metadata.  Copy images with detectable human poses into a /library folder and create json or other document with associated metadata
2. Determine how to store images and metadata so they can be quickly matched and displayed
3. Implement high-speed matching to display matches in near real time
4. Improve UX to display information from metadata with the image
5. Create a standard image page to provide key details about arbitrary images based on a unique identifier
6. Add accessibility features (maybe add alt text during the person_or_not phase, and slow down refresh rate to make it easier to process alt text display?  this would also be an opportunity to add the "about" menu)
7. Analytics
8. Confirm that browser can choose webcam
X - start here
9. Create script that either a) copies images and metadata from image_cleaner to the main folder, replacing existing images and metadata in the main folder, or b) copies the images and metadata from image_cleaner, appending them to the existing images and metadata in the main folder. For b, you probably want an option to add some sort of prefix to the object_ID and image file name (like 2 characters) to avoid namespace collisions
10. Create a new utility to download images from smithsonian
11. Use 9b to integrate smithsonian images into dataset
12. Add branding and styling to primary page in a way that can easily be restyled (maybe handed with a separate file, similarly to how analytics.js makes it easy to edit analytics)





### Phase 1 (working prototype)

1. create page that successfully identifies point detection for 3 images and accurately draws bounding boxes around them. Page allows user to toggle through the images with keypoints and bounding box displayed to confirm that the keypoints and bounding box are correct
2. for each image, a) resize and scale around a consistent square size, b) use L2 normalization for all of the keypoints, c) save the cosine similarity score for the normalized keypoints in an external file that can be referenced later
3. create page that tracks a user through a webcam. The page displays the user and the images with the closest cosine similarity side by side. It does this by a) capturing the user image every second, performing the resizing, scaling, normalization, and cosine similarity analysis used in step 2, b) comparing the resulting cosine similarity to those of the three images from step 2, and 3) displaying the correct image