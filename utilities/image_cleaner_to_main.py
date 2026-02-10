"""
Move processed images and metadata from image_cleaner to the main project folder.

Modes:
- replace: Clear existing input_images/ and input_image_metadata.json, then copy new data
- append:  Add new data to existing input_images/ and input_image_metadata.json
"""

import os
import json
import shutil
import glob

# ============== CONFIGURATION ==============

MODE = "append"  # "replace" or "append"

# ============== PATHS ==============
# All paths are relative to this script's location (utilities/)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_IMAGES_DIR = os.path.join(SCRIPT_DIR, "image_cleaner", "processed_images")
SOURCE_METADATA_FILE = os.path.join(SCRIPT_DIR, "image_cleaner", "image_metadata.json")
DEST_IMAGES_DIR = os.path.join(SCRIPT_DIR, "..", "input_images")
DEST_METADATA_FILE = os.path.join(SCRIPT_DIR, "..", "input_image_metadata.json")


# ============== FUNCTIONS ==============

def get_source_code():
    """Prompt the user for a 2-character source code."""
    while True:
        code = input("Enter a 2-character source code (e.g. 'MM' for Met Museum): ").strip()
        if len(code) == 2:
            return code
        print("Source code must be exactly 2 characters. Please try again.")


def load_json(filepath):
    """Load a JSON file and return its contents."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(filepath, data):
    """Save data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def prefix_metadata(records, code):
    """Prepend source code to object_id and filename in each record."""
    prefixed = []
    for record in records:
        new_record = record.copy()
        new_record['object_id'] = code + record['object_id']
        new_record['filename'] = code + record['filename']
        new_record['metadata'] = record['metadata'].copy()
        new_record['l2_vector'] = record['l2_vector'][:]
        prefixed.append(new_record)
    return prefixed


def replace_mode(source_records, code):
    """Replace all existing data with new data."""

    # Clear or create destination images folder
    if os.path.exists(DEST_IMAGES_DIR):
        for f in glob.glob(os.path.join(DEST_IMAGES_DIR, "*")):
            os.remove(f)
        print(f"Cleared {DEST_IMAGES_DIR}")
    else:
        os.makedirs(DEST_IMAGES_DIR)
        print(f"Created {DEST_IMAGES_DIR}")

    # Prefix the metadata
    prefixed_records = prefix_metadata(source_records, code)

    # Copy images with prefixed filenames
    copied = 0
    for record in prefixed_records:
        original_filename = record['filename'][len(code):]
        src_path = os.path.join(SOURCE_IMAGES_DIR, original_filename)
        dest_path = os.path.join(DEST_IMAGES_DIR, record['filename'])

        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            copied += 1
        else:
            print(f"  Warning: source image not found: {original_filename}")

    # Write metadata
    save_json(DEST_METADATA_FILE, prefixed_records)

    # Summary
    print("\n" + "=" * 50)
    print("REPLACE COMPLETE")
    print("=" * 50)
    print(f"Mode: replace")
    print(f"Source code: {code}")
    print(f"Images copied: {copied}")
    print(f"Total metadata records: {len(prefixed_records)}")


def append_mode(source_records, code):
    """Append new data to existing data."""

    # Load existing metadata (or start empty)
    if os.path.exists(DEST_METADATA_FILE):
        existing_records = load_json(DEST_METADATA_FILE)
        print(f"Loaded {len(existing_records)} existing records")
    else:
        existing_records = []
        print("No existing metadata found, starting fresh")

    # Create destination images folder if needed
    if not os.path.exists(DEST_IMAGES_DIR):
        os.makedirs(DEST_IMAGES_DIR)
        print(f"Created {DEST_IMAGES_DIR}")

    # Prefix the new metadata
    prefixed_records = prefix_metadata(source_records, code)

    # Check for duplicates
    existing_ids = {r['object_id'] for r in existing_records}
    new_records = []
    skipped = 0
    for record in prefixed_records:
        if record['object_id'] in existing_ids:
            skipped += 1
        else:
            new_records.append(record)

    if skipped > 0:
        print(f"  Skipping {skipped} duplicate records (object_id already exists)")

    # Copy new images with prefixed filenames
    copied = 0
    for record in new_records:
        original_filename = record['filename'][len(code):]
        src_path = os.path.join(SOURCE_IMAGES_DIR, original_filename)
        dest_path = os.path.join(DEST_IMAGES_DIR, record['filename'])

        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            copied += 1
        else:
            print(f"  Warning: source image not found: {original_filename}")

    # Merge and write metadata
    merged_records = existing_records + new_records
    save_json(DEST_METADATA_FILE, merged_records)

    # Summary
    print("\n" + "=" * 50)
    print("APPEND COMPLETE")
    print("=" * 50)
    print(f"Mode: append")
    print(f"Source code: {code}")
    print(f"New images copied: {copied}")
    print(f"Duplicates skipped: {skipped}")
    print(f"Total metadata records: {len(merged_records)}")


# ============== MAIN ==============

if __name__ == "__main__":
    # Validate mode
    if MODE not in ("replace", "append"):
        print(f"Error: MODE must be 'replace' or 'append', got '{MODE}'")
        exit(1)

    # Validate source files exist
    if not os.path.exists(SOURCE_METADATA_FILE):
        print(f"Error: source metadata not found: {SOURCE_METADATA_FILE}")
        exit(1)

    if not os.path.exists(SOURCE_IMAGES_DIR):
        print(f"Error: source images folder not found: {SOURCE_IMAGES_DIR}")
        exit(1)

    # Load source metadata
    source_records = load_json(SOURCE_METADATA_FILE)
    print(f"Loaded {len(source_records)} records from image_cleaner")

    # Get source code
    code = get_source_code()

    # Confirm before replace
    if MODE == "replace" and os.path.exists(DEST_METADATA_FILE):
        confirm = input(f"Replace mode will delete all existing data. Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            exit(0)

    # Run
    if MODE == "replace":
        replace_mode(source_records, code)
    else:
        append_mode(source_records, code)
