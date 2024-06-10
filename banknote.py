from google.colab import drive
drive.mount('/content/drive')
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at {path} could not be loaded.")
    return image

def extract_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def main():
    reference_image_dir = '/content/drive/MyDrive/reference_images'  # Update this path
    query_image_path = '/content/drive/MyDrive/query_banknote.jpg'  # Update this path

    query_image = load_image(query_image_path)
    kp_query, des_query = extract_features(query_image)

    best_match_image = None
    best_match_count = 0
    best_matches = None
    best_kp_ref = None
    best_kp_query = None
    best_ref_image_name = None

    for ref_image_name in os.listdir(reference_image_dir):
        ref_image_path = os.path.join(reference_image_dir, ref_image_name)
        reference_image = load_image(ref_image_path)
        kp_ref, des_ref = extract_features(reference_image)

        matches = match_features(des_query, des_ref)

        if len(matches) > best_match_count:
            best_match_count = len(matches)
            best_match_image = reference_image
            best_matches = matches
            best_kp_ref = kp_ref
            best_kp_query = kp_query
            best_ref_image_name = ref_image_name

    if best_match_image is not None:
        result_image = cv2.drawMatches(query_image, best_kp_query, best_match_image, best_kp_ref, best_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Add text to the image
        if best_ref_image_name is not None:
            text_position = (10, 30)
            cv2.putText(result_image, best_ref_image_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display the result using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(result_image)
        plt.title('Best Match')
        plt.axis('off')  # Hide axes
        plt.show()
    else:
        print("No matches found.")

if __name__ == "__main__":
    main()
