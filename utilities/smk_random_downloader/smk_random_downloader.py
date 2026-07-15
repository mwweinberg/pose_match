import requests
import time
import random
import shutil
import os
import json

from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

number_of_pictures_to_process = 300
number_of_pictures_processed = 0

#so you know how long it takes to run this thing
start_time = time.time()

# SMK API endpoint (no API key required)
API_BASE_URL = "https://api.smk.dk/api/v1/art/search/"

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


##########SMK API HELPER FUNCTIONS##########

def get_total_count():
    """Get total number of artworks matching our filters."""
    params = {
        'keys': '*',
        'filters': '[has_image:true],[public_domain:true]',
        'offset': 0,
        'rows': 0,
        'lang': 'en'
    }
    response = requests.get(API_BASE_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    return data.get('found', 0)


def get_title(item):
    """Extract title from the titles array."""
    titles = item.get('titles', [])
    if titles:
        return titles[0].get('title', '')
    return ''


def get_artist(item):
    """Extract artist name from the artist array or production array."""
    # The simplified artist array is easiest
    artists = item.get('artist', [])
    if artists:
        return artists[0]
    # Fall back to production array
    production = item.get('production', [])
    if production:
        return production[0].get('creator', '')
    return ''


def get_date(item):
    """Extract date from production_date array."""
    dates = item.get('production_date', [])
    if dates:
        return dates[0].get('period', '')
    return ''


def get_medium(item):
    """Extract medium/technique."""
    techniques = item.get('techniques', [])
    if techniques:
        return ', '.join(techniques)
    return ''


def get_object_type(item):
    """Extract object type from object_names array."""
    names = item.get('object_names', [])
    if names:
        return names[0].get('name', '')
    return ''


##########END SMK API HELPER FUNCTIONS######


# Get total count for random offset generation
print("Querying SMK API for total artwork count...")
total_count = get_total_count()
print(f"Found {total_count} artworks with images in the public domain")

images_downloaded = 0
used_object_numbers = set()
metadata_holder = []

print(f"Target: {number_of_pictures_to_process} images with valid poses")

#start looping as long as the number of pictures you download is less than the number of pictures you want to download
while number_of_pictures_processed < number_of_pictures_to_process:

    # Pick a random offset and fetch a batch
    random_offset = random.randint(0, max(0, total_count - 10))
    params = {
        'keys': '*',
        'filters': '[has_image:true],[public_domain:true]',
        'offset': random_offset,
        'rows': 10,
        'lang': 'en'
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
    items = data.get('items', [])

    if not items:
        print('No results returned, retrying...')
        time.sleep(2)
        continue

    # Process each result in the batch
    for item in items:
        if number_of_pictures_processed >= number_of_pictures_to_process:
            break

        object_number = item.get('object_number', '')

        # Skip if already processed
        if object_number in used_object_numbers:
            print(f'Already processed {object_number}, skipping')
            continue

        used_object_numbers.add(object_number)

        # Get image URL (thumbnail is ~1024px JPEG, plenty for pose detection)
        object_image_url = item.get('image_thumbnail', '')
        if not object_image_url:
            print(f'No image URL for {object_number}')
            continue

        # Extract metadata
        object_title = get_title(item)
        object_artist = get_artist(item)
        object_date = get_date(item)
        object_type = get_object_type(item)
        object_medium = get_medium(item)
        object_link = item.get('frontend_url', '')

        #just keeping track
        images_downloaded += 1

        print(f'Downloading image from {object_image_url}')

        #download the file
        try:
            r = requests.get(object_image_url, timeout=15)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f'Skipping {object_number}: {e}')
            continue
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f'Connection error downloading {object_number}: {e}')
            print('Retrying in 10 seconds...')
            time.sleep(10)
            try:
                r = requests.get(object_image_url, timeout=15)
                r.raise_for_status()
            except Exception as e2:
                print(f'Retry failed, skipping {object_number}: {e2}')
                continue

        #save it as temp_image.jpg
        with open("temp_image.jpg", "wb") as f:
            f.write(r.content)

        # Convert to JPEG (safety net for non-JPEG formats)
        try:
            img = Image.open("temp_image.jpg")
            img = img.convert("RGB")
            img.save("temp_image.jpg", "JPEG")
        except Exception as e:
            print(f'  Could not convert image for {object_number}: {e}')
            continue

        #process the image with tensorflow
        keypoints = detect_pose(movenet, "temp_image.jpg")
        # Check if valid pose
        if is_valid_pose(keypoints):
            print('person!')
            print(f'{number_of_pictures_processed+1} of {number_of_pictures_to_process} saved.')

            #move & rename the file (sanitize slashes in object numbers like "KKS14612/16")
            safe_filename = str(object_number).replace("/", "_")
            shutil.move("temp_image.jpg", "output_images/"+safe_filename+".jpg")

            image_metadata = {
                "Object_ID": safe_filename,
                'object_image_url': object_image_url,
                'Title': object_title,
                'Artist_Display_Name': object_artist,
                'Object_Date': object_date,
                'Department': object_type,
                'Medium': object_medium,
                'Link_Resource': object_link,
                'Repository': 'SMK - Statens Museum for Kunst'
                }

            metadata_holder.append(image_metadata)

            number_of_pictures_processed += 1

        else:
            print(f'not a person! still only {number_of_pictures_processed} of {number_of_pictures_to_process} saved.')

    # Brief pause between batches
    time.sleep(1)


with open("cleaned_met_data.json", "w") as f:
    json.dump(metadata_holder, f, indent=2)


#so you know how long it takes to run this thing
end_time = time.time()
#default is in seconds, so divide by 60 to get minutes
elapsed_time = (end_time - start_time) / 60

print(f'Tested {len(used_object_numbers)} objects. Downloaded {images_downloaded} images, {number_of_pictures_processed} of which were people.')

print(f'It took {elapsed_time} minutes to run')
