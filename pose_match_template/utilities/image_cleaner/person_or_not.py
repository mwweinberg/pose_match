"""
Batch process images to identify those with detectable human poses.
Uses TensorFlow MoveNet (same model as ml5.js) for consistency.

Outputs:
- /processed_images/ folder with filtered images
- image_metadata.json with metadata + L2 vectors for matching
"""

import os
import json
import shutil
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# ============== CONFIGURATION ==============

# Paths
INPUT_IMAGES_FOLDER = "input_images"
OUTPUT_IMAGES_FOLDER = "processed_images"
INPUT_METADATA_FILE = "input_metadata.json"
OUTPUT_METADATA_FILE = "image_metadata.json"

# Pose detection thresholds
# Minimum average confidence across all keypoints to consider a pose "detected"
MIN_AVERAGE_CONFIDENCE = 0.3
# Minimum number of keypoints that must exceed confidence threshold
MIN_CONFIDENT_KEYPOINTS = 10
# Confidence threshold for individual keypoints
KEYPOINT_CONFIDENCE_THRESHOLD = 0.1

# MoveNet model URL (SinglePose Lightning - same as ml5.js default)
MOVENET_MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

# MoveNet keypoint names (same order as ml5.js)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


# ============== NORMALIZATION FUNCTIONS ==============
# These match the JavaScript functions in sketch.js exactly

def compute_bounding_box(keypoints, confidence_threshold=KEYPOINT_CONFIDENCE_THRESHOLD):
    """Compute bounding box from keypoints with confidence threshold."""
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')

    for kp in keypoints:
        if kp['confidence'] > confidence_threshold:
            x_min = min(x_min, kp['x'])
            x_max = max(x_max, kp['x'])
            y_min = min(y_min, kp['y'])
            y_max = max(y_max, kp['y'])

    return {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}


def normalize_keypoints(keypoints, confidence_threshold=KEYPOINT_CONFIDENCE_THRESHOLD):
    """Normalize keypoints: center around bounding box and scale to unit square."""
    box = compute_bounding_box(keypoints, confidence_threshold)

    # Compute center and size of bounding box
    center_x = (box['x_min'] + box['x_max']) / 2
    center_y = (box['y_min'] + box['y_max']) / 2
    box_width = box['x_max'] - box['x_min']
    box_height = box['y_max'] - box['y_min']

    # Use the larger dimension to maintain aspect ratio
    scale = max(box_width, box_height)

    # Avoid division by zero
    if scale == 0:
        scale = 1

    # Normalize each keypoint
    normalized = []
    for kp in keypoints:
        normalized.append({
            'name': kp['name'],
            'x': (kp['x'] - center_x) / scale,
            'y': (kp['y'] - center_y) / scale,
            'confidence': kp['confidence']
        })

    return normalized


def keypoints_to_vector(keypoints):
    """Flatten keypoints to a vector [x1, y1, x2, y2, ...] for L2 normalization."""
    vector = []
    for kp in keypoints:
        vector.append(kp['x'])
        vector.append(kp['y'])
    return vector


def l2_normalize(vector):
    """L2 normalize a vector (divide by magnitude to create unit vector)."""
    sum_of_squares = sum(v * v for v in vector)
    magnitude = sum_of_squares ** 0.5

    # Avoid division by zero
    if magnitude == 0:
        return vector

    return [v / magnitude for v in vector]


def process_pose(keypoints):
    """Process keypoints: normalize and apply L2 normalization."""
    normalized_keypoints = normalize_keypoints(keypoints)
    vector = keypoints_to_vector(normalized_keypoints)
    l2_vector = l2_normalize(vector)

    return {
        'normalized_keypoints': normalized_keypoints,
        'l2_vector': l2_vector
    }


# ============== POSE DETECTION FUNCTIONS ==============

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


# ============== MAIN PROCESSING FUNCTION ==============

def load_metadata():
    """Load the Met metadata JSON file."""
    print(f"Loading metadata from {INPUT_METADATA_FILE}...")
    with open(INPUT_METADATA_FILE, 'r') as f:
        metadata_list = json.load(f)

    # Convert to dictionary keyed by Object_ID for easy lookup
    metadata_dict = {}
    for item in metadata_list:
        object_id = str(item.get('Object_ID', ''))
        if object_id:
            metadata_dict[object_id] = item

    print(f"Loaded metadata for {len(metadata_dict)} objects.")
    return metadata_dict


def process_all_images():
    """Main function to process all images and filter those with valid poses."""

    # Load the MoveNet model
    movenet = load_model()

    # Load metadata
    metadata_dict = load_metadata()

    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_IMAGES_FOLDER):
        os.makedirs(OUTPUT_IMAGES_FOLDER)
        print(f"Created output folder: {OUTPUT_IMAGES_FOLDER}")

    # Get list of image files
    image_files = [f for f in os.listdir(INPUT_IMAGES_FOLDER)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    print(f"Found {total_images} images to process.")

    # Track results
    processed_count = 0
    valid_pose_count = 0
    failed_count = 0
    output_metadata = []

    # Process each image
    for i, filename in enumerate(image_files):
        # Progress update every 50 images
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Processing image {i + 1}/{total_images}...")

        image_path = os.path.join(INPUT_IMAGES_FOLDER, filename)

        # Extract Object_ID from filename (assumes format: object_id.jpg)
        object_id = os.path.splitext(filename)[0]

        # Detect pose
        keypoints = detect_pose(movenet, image_path)
        processed_count += 1

        # Check if valid pose
        if is_valid_pose(keypoints):
            valid_pose_count += 1

            # Process pose to get L2 vector
            pose_data = process_pose(keypoints)

            # Copy image to output folder
            output_path = os.path.join(OUTPUT_IMAGES_FOLDER, filename)
            shutil.copy2(image_path, output_path)

            # Get metadata for this image
            image_metadata = metadata_dict.get(object_id, {})

            # Create output record with metadata and pose data
            output_record = {
                'object_id': object_id,
                'filename': filename,
                'l2_vector': pose_data['l2_vector'],
                'metadata': {
                    'title': image_metadata.get('Title', ''),
                    'artist': image_metadata.get('Artist_Display_Name', ''),
                    'date': image_metadata.get('Object_Date', ''),
                    'department': image_metadata.get('Department', ''),
                    'medium': image_metadata.get('Medium', ''),
                    'link': image_metadata.get('Link_Resource', ''),
                    'repository': image_metadata.get('Repository', '')
                }
            }
            output_metadata.append(output_record)

        elif keypoints is None:
            failed_count += 1

    # Save output metadata
    print(f"\nSaving metadata to {OUTPUT_METADATA_FILE}...")
    with open(OUTPUT_METADATA_FILE, 'w') as f:
        json.dump(output_metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Total images processed: {processed_count}")
    print(f"Images with valid poses: {valid_pose_count}")
    print(f"Images without valid poses: {processed_count - valid_pose_count - failed_count}")
    print(f"Images that failed to load: {failed_count}")
    print(f"\nOutput saved to:")
    print(f"  - Images: {OUTPUT_IMAGES_FOLDER}/")
    print(f"  - Metadata: {OUTPUT_METADATA_FILE}")


# ============== ENTRY POINT ==============

#using 'if __name__...' allows you to import this script as a library and call functions within it
#so, for example, you would 'import person_or_not' and then call 'person_or_not.is_valid_pose()
if __name__ == "__main__":
    process_all_images()