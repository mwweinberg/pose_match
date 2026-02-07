import csv
import requests
import time 
import random
import shutil
import os
import json

import tensorflow as tf
import tensorflow_hub as hub

number_of_pictures_to_process = 500
number_of_pictures_processed = 0

#so you know how long it takes to run this thing
start_time = time.time()

INPUT_METADATA_FILE = "MetObjects.csv"

#create the output directory for the matched images
os.makedirs("output_images", exist_ok=True)


##########TENSORFLOW VARIABLES AND FUNCTIONS##########

# MoveNet model URL (SinglePose Lightning - same as ml5.js default)
MOVENET_MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
# MoveNet keypoint names (same order as ml5.js)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
# Confidence threshold for individual keypoints
KEYPOINT_CONFIDENCE_THRESHOLD = 0.2
# Minimum number of keypoints that must exceed confidence threshold
MIN_CONFIDENT_KEYPOINTS = 12
# Minimum average confidence across all keypoints to consider a pose "detected"
MIN_AVERAGE_CONFIDENCE = 0.4

def load_model():
    """Load MoveNet model from TensorFlow Hub."""
    print("Loading MoveNet model...")
    model = hub.load(MOVENET_MODEL_URL)
    movenet = model.signatures['serving_default']
    print("Model loaded successfully.")
    return movenet




def load_and_preprocess_image(image_path):
    """Load and preprocess image for MoveNet (expects 192x192 for Lightning)."""
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Store original dimensions for scaling keypoints back
    original_height = tf.shape(image)[0]
    original_width = tf.shape(image)[1]

    # Resize to 192x192 (MoveNet Lightning input size)
    image = tf.image.resize_with_pad(image, 192, 192)
    image = tf.cast(image, dtype=tf.int32)

    # Add batch dimension: model expects shape [1, 192, 192, 3] not [192, 192, 3]
    input_image = tf.expand_dims(image, axis=0)

    return input_image, int(original_width), int(original_height)


def detect_pose(movenet, image_path):
    """Run pose detection on an image and return keypoints."""
    try:
        input_image, original_width, original_height = load_and_preprocess_image(image_path)

        # Run inference
        outputs = movenet(input_image)
        keypoints_with_scores = outputs['output_0'].numpy()[0, 0, :, :]

        # Convert to our keypoint format (matching ml5.js structure)
        # MoveNet returns [y, x, confidence] normalized to 0-1
        keypoints = []
        for i, name in enumerate(KEYPOINT_NAMES):
            y, x, confidence = keypoints_with_scores[i]
            keypoints.append({
                'name': name,
                'x': float(x * original_width),   # Scale to original image dimensions
                'y': float(y * original_height),
                'confidence': float(confidence)
            })

        return keypoints

    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return None


def is_valid_pose(keypoints):
    """Check if the detected pose meets our quality thresholds."""
    if keypoints is None:
        return False

    # Count keypoints above confidence threshold
    confident_keypoints = sum(
        1 for kp in keypoints
        if kp['confidence'] > KEYPOINT_CONFIDENCE_THRESHOLD
    )

    # Calculate average confidence
    avg_confidence = sum(kp['confidence'] for kp in keypoints) / len(keypoints)

    # Check both criteria
    meets_count = confident_keypoints >= MIN_CONFIDENT_KEYPOINTS
    meets_avg = avg_confidence >= MIN_AVERAGE_CONFIDENCE

    return meets_count and meets_avg

movenet = load_model()

##########END TENSORFLOW FUNCTIONS######


#####load full met CSV as an object
met_metadata = []

print(f"Loading metadata from {INPUT_METADATA_FILE}...")
with open(INPUT_METADATA_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        met_metadata.append(row)
    
   

#### get the number of objects in the CSV
number_of_objects = len(met_metadata)
print(f"metadata has {number_of_objects} rows")

print(met_metadata[53]['Object ID'])

images_downloaded = 0
used_random_numbers = []
metadata_holder = []

#start looping as long as the number of pictures you download is less than the number of pictures you want to download
while number_of_pictures_processed <= number_of_pictures_to_process:
    this_object_ID = met_metadata[random.randint(1, number_of_objects)]['Object ID']
      
    #random is truly random, so this is to make sure that you don't repeat object_IDs
    if this_object_ID in used_random_numbers:
        print(f'random means random! we have a repeat of {this_object_ID}')
    else:
        #construct the URL
        met_api_url = 'https://collectionapi.metmuseum.org/public/collection/v1/objects/'+this_object_ID
        #get the data
        try:
            response = requests.get(met_api_url)
            response.raise_for_status()
        #if there is an error in the url, this round of the while loop will end 
        except requests.exceptions.HTTPError as e:
            print(f'Skipping {this_object_ID}: {e}')
            used_random_numbers.append(this_object_ID)
            continue
        #convert into dict
        object_data = response.json()

        #print(object_data)

        object_ID = object_data['objectID']
        object_image_url = object_data['primaryImage']
        object_title = object_data['title']
        object_artist = object_data['artistDisplayName']
        object_date = object_data['objectDate']
        object_department = object_data['department']
        object_medium = object_data['medium']
        object_link = object_data['objectURL']
        object_repository = object_data['repository']

        if object_image_url:

            #just keeping track
            images_downloaded += 1

            print(f'Downloading image from {object_image_url}')

            #download the file
            try:
                r = requests.get(object_image_url, timeout=15)
                r.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print(f'Skipping {this_object_ID}: {e}')
                used_random_numbers.append(this_object_ID)
                continue

            #save it as temp_image.jpg (note, things will be bad if you use this somewhere else that does not use jpg...)
            with open("temp_image.jpg", "wb") as f:
                f.write(r.content)

            


            #process the image with tensorflow
            # Detect pose
            #keypoints = detect_pose(movenet, image_path)
            keypoints = detect_pose(movenet, "temp_image.jpg")
            # Check if valid pose
            if is_valid_pose(keypoints):
                print('person!')
                print(f'{number_of_pictures_processed} of {number_of_pictures_to_process} saved.')

                #move & rename the file
                shutil.move("temp_image.jpg", "output_images/"+str(object_ID)+".jpg")

                

                image_metadata = {
                    "Object_ID": object_ID,
                    'object_image_url': object_image_url,
                    'Title': object_title,
                    'Artist_Display_Name': object_artist,
                    'Object_Date': object_date,
                    'Department': object_department,
                    'Medium': object_medium,
                    'Link_Resource': object_link,
                    'Repository': object_repository
                    }

                metadata_holder.append(image_metadata)

                number_of_pictures_processed += 1

            else:
                print(f'not a person! still only {number_of_pictures_processed} of {number_of_pictures_to_process} saved.')








            
        else:
            print('no object url')
        used_random_numbers.append(this_object_ID)
        time.sleep(0.1)


#TODO: write metadata object to json

with open("cleaned_met_data.json", "w") as f:
    json.dump(metadata_holder, f, indent=2)


print(used_random_numbers)
print(metadata_holder)


#so you know how long it takes to run this thing
end_time = time.time()
#default is in seconds, so divide by 60 to get minutes
elapsed_time = (end_time - start_time) / 60  

print(f'Tested {len(used_random_numbers)} objects. Downloaded {images_downloaded} images, {number_of_pictures_processed} of which were people.')
        



#print(object_image_url)



    


