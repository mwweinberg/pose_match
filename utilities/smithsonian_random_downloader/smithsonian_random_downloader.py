import requests
import time
import shutil
import os
import json

import tensorflow as tf
import tensorflow_hub as hub

number_of_pictures_to_process = 500
number_of_pictures_processed = 0

#so you know how long it takes to run this thing
start_time = time.time()

# Load API key from config.json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(SCRIPT_DIR, "config.json")
if not os.path.exists(config_path):
    print("Error: config.json not found. Copy config_example.json to config.json and add your API key.")
    print("Get a free API key at: https://api.data.gov/signup/")
    exit(1)

with open(config_path, 'r') as f:
    config = json.load(f)

API_KEY = config['api_key']
if API_KEY == "YOUR_API_KEY_HERE":
    print("Error: Please add your API key to config.json")
    print("Get a free API key at: https://api.data.gov/signup/")
    exit(1)

# Smithsonian API endpoint (art_design category to focus on art collections)
API_BASE_URL = "https://api.si.edu/openaccess/api/v1.0/category/art_design/search"

# Unit codes to exclude (NMNH collections and National Zoo)
EXCLUDED_UNIT_CODES = {
    "NMNHANTHRO", "NMNHBIRDS", "NMNHBOTANY", "NMNHENTO",
    "NMNHFISHES", "NMNHHERPS", "NMNHINV", "NMNHMAMMALS",
    "NMNHMINSCI", "NMNHPALEO", "NZP"
}

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


##########SMITHSONIAN API HELPER FUNCTIONS##########

def extract_field_from_freetext(freetext, field_name, label_filter=None):
    """Extract a value from the freetext section of the API response.
    If label_filter is provided, only return values where the label matches."""
    if field_name not in freetext:
        return ""
    entries = freetext[field_name]
    if not entries:
        return ""
    if label_filter:
        for entry in entries:
            if entry.get('label', '').lower() == label_filter.lower():
                return entry.get('content', '')
        return ""
    return entries[0].get('content', '')


def get_image_url(row):
    """Extract the image download URL from a Smithsonian API result."""
    try:
        online_media = row['content']['descriptiveNonRepeating']['online_media']
        media = online_media['media']
        for item in media:
            if item.get('type') == 'Images':
                # Prefer high-res JPEG from resources
                resources = item.get('resources', [])
                for resource in resources:
                    if 'High-resolution' in resource.get('label', ''):
                        return resource['url']
                # Fall back to content URL
                return item.get('content', '')
        return ""
    except (KeyError, IndexError, TypeError):
        return ""


def get_artist(freetext):
    """Extract artist name from freetext, checking common label variations."""
    if 'name' not in freetext:
        return ""
    for entry in freetext['name']:
        label = entry.get('label', '').lower()
        if label in ('artist', 'maker', 'creator', 'designer', 'author', 'painter', 'sculptor'):
            return entry.get('content', '')
    # If no artist-specific label found, return first name entry
    return freetext['name'][0].get('content', '') if freetext['name'] else ""


##########END SMITHSONIAN API HELPER FUNCTIONS######


images_downloaded = 0
used_record_ids = set()
metadata_holder = []

print(f"Target: {number_of_pictures_to_process} images with valid poses")
print(f"Excluded collections: NMNH (all), National Zoo")

#start looping as long as the number of pictures you download is less than the number of pictures you want to download
while number_of_pictures_processed < number_of_pictures_to_process:

    # Fetch a batch of random results from the Smithsonian API
    params = {
        'q': '*:*',
        'rows': 10,
        'sort': 'random',
        'api_key': API_KEY,
        'fq': 'online_media_type:Images'
    }

    try:
        response = requests.get(API_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f'API error: {e}')
        time.sleep(5)
        continue
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        print(f'Connection error: {e}')
        print('Retrying in 10 seconds...')
        time.sleep(10)
        try:
            response = requests.get(API_BASE_URL, params=params, timeout=15)
            response.raise_for_status()
        except Exception as e2:
            print(f'Retry failed: {e2}')
            continue

    data = response.json()
    rows = data.get('response', {}).get('rows', [])

    if not rows:
        print('No results returned, retrying...')
        time.sleep(2)
        continue

    # Process each result in the batch
    for row in rows:
        if number_of_pictures_processed >= number_of_pictures_to_process:
            break

        record_id = row.get('content', {}).get('descriptiveNonRepeating', {}).get('record_ID', '')
        unit_code = row.get('unitCode', '')

        # Skip if already processed
        if record_id in used_record_ids:
            print(f'Already processed {record_id}, skipping')
            continue

        used_record_ids.add(record_id)

        # Skip excluded collections
        if unit_code in EXCLUDED_UNIT_CODES:
            print(f'Skipping {record_id} from excluded collection: {unit_code}')
            continue

        # Get image URL
        object_image_url = get_image_url(row)
        if not object_image_url:
            print(f'No image URL for {record_id}')
            continue

        # Extract metadata
        freetext = row.get('content', {}).get('freetext', {})
        descriptive = row.get('content', {}).get('descriptiveNonRepeating', {})

        object_title = row.get('title', '')
        object_artist = get_artist(freetext)
        object_date = extract_field_from_freetext(freetext, 'date')
        object_department = descriptive.get('data_source', '')
        object_medium = extract_field_from_freetext(freetext, 'physicalDescription', label_filter='Medium')
        object_link = descriptive.get('record_link', '')
        object_repository = descriptive.get('data_source', '')

        #just keeping track
        images_downloaded += 1

        print(f'Downloading image from {object_image_url}')

        #download the file
        try:
            r = requests.get(object_image_url, timeout=15)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f'Skipping {record_id}: {e}')
            continue
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f'Connection error downloading {record_id}: {e}')
            print('Retrying in 10 seconds...')
            time.sleep(10)
            try:
                r = requests.get(object_image_url, timeout=15)
                r.raise_for_status()
            except Exception as e2:
                print(f'Retry failed, skipping {record_id}: {e2}')
                continue

        #save it as temp_image.jpg
        with open("temp_image.jpg", "wb") as f:
            f.write(r.content)

        #process the image with tensorflow
        keypoints = detect_pose(movenet, "temp_image.jpg")
        # Check if valid pose
        if is_valid_pose(keypoints):
            print('person!')
            print(f'{number_of_pictures_processed+1} of {number_of_pictures_to_process} saved.')

            #move & rename the file
            shutil.move("temp_image.jpg", "output_images/"+str(record_id)+".jpg")

            image_metadata = {
                "Object_ID": record_id,
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

    # Respect rate limit: ~4 seconds between batches keeps us well under 1000 requests/hour
    time.sleep(4)


with open("cleaned_met_data.json", "w") as f:
    json.dump(metadata_holder, f, indent=2)


#so you know how long it takes to run this thing
end_time = time.time()
#default is in seconds, so divide by 60 to get minutes
elapsed_time = (end_time - start_time) / 60

print(f'Tested {len(used_record_ids)} objects. Downloaded {images_downloaded} images, {number_of_pictures_processed} of which were people.')

print(f'It took {elapsed_time} minutes to run')
