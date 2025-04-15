import os
import cv2
import numpy as np
import logging
import pickle
import face_recognition  # More accurate than raw MediaPipe landmarks
from pathlib import Path
from datetime import datetime

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
EMBEDDINGS_FILE = BASE_DIR / "face_embeddings.pkl"
DEBUG_DIR = BASE_DIR / "debug_images"  # For saving debug visualizations
DEBUG_DIR.mkdir(exist_ok=True)

def get_face_encodings(image_path, save_debug=False):
    """
    Extract face encodings from an image using face_recognition library.
    Returns a list of face encodings for each detected face.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"⚠️ Unable to load image: {image_path}")
        return [], None
    
    # Convert BGR to RGB (face_recognition uses RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    if not face_locations:
        logger.info(f"ℹ️ No faces detected in {image_path}")
        return [], None
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    # Save debug image if requested
    debug_image = None
    if save_debug and face_locations:
        debug_image = image.copy()
        for (top, right, bottom, left) in face_locations:
            # Draw a box around the face
            cv2.rectangle(debug_image, (left, top), (right, bottom), (0, 255, 0), 2)
    
    return face_encodings, debug_image

def generate_embeddings_database():
    """
    Generates and saves face encodings for all known faces.
    Directory structure: known_faces/<domain>/<name>.jpg
    
    Note: Each image filename should match the full_name in the user_profiles table
    for proper user identification.
    """
    database = {
        "encodings": [],
        "names": [],
        "domains": [],
        "timestamp": datetime.now().isoformat(),
        "image_paths": []  # Store image paths for reference and debugging
    }
    
    if not KNOWN_FACES_DIR.exists():
        logger.warning("⚠️ known_faces directory is missing!")
        return database
        
    logger.info(f"📁 Generating encodings database from: {KNOWN_FACES_DIR}")
    
    # Process each domain directory
    skipped_count = 0
    for domain_dir in KNOWN_FACES_DIR.iterdir():
        if not domain_dir.is_dir():
            continue
            
        domain = domain_dir.name
        logger.info(f"📂 Processing domain: {domain}")
        
        # Process each image file in this domain
        for img_path in domain_dir.glob("*.jpg"):
            if not img_path.is_file():
                continue
                
            # Extract person name from filename (remove any parentheses content)
            # This name should match exactly with the full_name in user_profiles table
            person_name = img_path.stem.split('(')[0].strip()
            
            try:
                # Get face encodings and debug image
                encodings, debug_image = get_face_encodings(img_path, save_debug=True)
                
                if not encodings:
                    logger.warning(f"⚠️ No face detected in {img_path.name}")
                    skipped_count += 1
                    continue
                
                if len(encodings) > 1:
                    logger.warning(f"⚠️ Multiple faces ({len(encodings)}) found in {img_path.name}, using the first one")
                
                # Save the debug image
                debug_path = DEBUG_DIR / f"{domain}_{person_name}_debug.jpg"
                if debug_image is not None:
                    cv2.imwrite(str(debug_path), debug_image)
                
                # Add first encoding to database (assuming one person per image)
                database["encodings"].append(encodings[0])
                database["names"].append(person_name)
                database["domains"].append(domain)
                database["image_paths"].append(str(img_path))
                
                logger.info(f"✅ Added encoding for {person_name} in {domain}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {img_path.name}: {str(e)}")
    
    # Save the database to a pickle file
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(database, f)
    
    logger.info(f"💾 Saved encodings database with {len(database['names'])} faces")
    logger.info(f"🔴 Skipped {skipped_count} images with no detectable faces")
    
    return database

def load_embeddings_database(force_refresh=False):
    """
    Loads face encodings database from pickle file or generates it if needed.
    """
    if EMBEDDINGS_FILE.exists() and not force_refresh:
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                database = pickle.load(f)
                logger.info(f"📊 Loaded encodings database with {len(database['names'])} faces")
                return database
        except Exception as e:
            logger.error(f"❌ Error loading encodings file: {e}")
            logger.info("🔄 Regenerating database due to loading error")
    
    # If loading failed, file doesn't exist, or force refresh is requested
    return generate_embeddings_database()

def verify_faces(test_encodings, known_database, tolerance=0.5):
    """
    Verifies if any face in test_encodings matches with known faces.
    
    Args:
        test_encodings: List of face encodings from the test image
        known_database: Dictionary containing known face data
        tolerance: Lower tolerance means more strict matching (0.4-0.6 is a good range)
        
    Returns:
        List of tuples (name, domain, confidence) for matched faces
    """
    if not test_encodings or not known_database["encodings"]:
        return []
    
    # Convert to numpy arrays for vectorized operations
    known_encodings = np.array(known_database["encodings"])
    known_names = known_database["names"]
    known_domains = known_database["domains"]
    
    # Store matched faces
    matched_faces = []
    seen_names = set()
    
    for test_encoding in test_encodings:
        # Calculate face distances (lower distance = better match)
        face_distances = face_recognition.face_distance(known_encodings, test_encoding)
        
        # Find the best match (lowest distance)
        best_match_idx = np.argmin(face_distances)
        best_match_distance = face_distances[best_match_idx]
        
        # Convert distance to a similarity score (0-1 where 1 is perfect match)
        # 0.6 distance is roughly a 40% match, 0 distance is 100% match
        similarity_score = max(0, 1 - best_match_distance * 1.67)
        
        # Only accept match if distance is below tolerance
        if best_match_distance <= tolerance:
            name = known_names[best_match_idx]
            domain = known_domains[best_match_idx]
            
            # Avoid duplicates (only return the best match for each name)
            if name not in seen_names:
                seen_names.add(name)
                matched_faces.append((name, domain, similarity_score))
                logger.info(f"✅ Match found: {name} ({domain}) - similarity: {similarity_score:.2f}")
        else:
            logger.debug(f"ℹ️ Face detected but no match found (best distance: {best_match_distance:.3f})")
    
    # Sort by confidence (highest first)
    matched_faces.sort(key=lambda x: x[2], reverse=True)
    return matched_faces

def recognize_faces(image_path, force_refresh=False):
    """
    Recognizes faces in an image and returns names, domains and confidence scores.
    """
    try:
        # Load or generate encodings database
        database = load_embeddings_database(force_refresh)
        
        # Extract encodings from the input image
        logger.info(f"🔍 Analyzing image: {image_path}")
        test_encodings, debug_image = get_face_encodings(image_path, save_debug=True)
        
        if not test_encodings:
            logger.warning("⚠️ No faces found in uploaded image")
            return [], [], []
        
        # Save debug image with face boxes
        if debug_image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = DEBUG_DIR / f"upload_{timestamp}_debug.jpg"
            cv2.imwrite(str(debug_path), debug_image)
            logger.info(f"📷 Debug image saved to {debug_path}")
        
        # Match faces against database
        matched_faces = verify_faces(test_encodings, database)
        
        # Extract results
        names = [match[0] for match in matched_faces]
        domains = [match[1] for match in matched_faces]
        confidences = [match[2] for match in matched_faces]
        
        if matched_faces:
            logger.info(f"✅ Found {len(matched_faces)} matches in image")
        else:
            logger.warning("⚠️ No matches found for any faces in the image")
            
        return names, domains, confidences
                
    except Exception as e:
        logger.error(f"❌ Error in recognize_faces: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return [], [], []

# Main function to test the module independently
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Recognition Tool")
    parser.add_argument("--update-database", action="store_true", help="Update face embeddings database")
    parser.add_argument("--test", type=str, help="Test image path")
    parser.add_argument("--force", action="store_true", help="Force database refresh")
    
    args = parser.parse_args()
    
    # Regenerate embeddings database if requested
    if args.update_database:
        generate_embeddings_database()
        sys.exit(0)
        
    # Test with an image if provided
    if args.test:
        names, domains, confidences = recognize_faces(args.test, force_refresh=args.force)
        
        if names:
            print(f"Recognized {len(names)} people:")
            for name, domain, confidence in zip(names, domains, confidences):
                print(f"- {name} ({domain}) - confidence: {confidence:.2f}")
        else:
            print("No matches found")