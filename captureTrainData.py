"""
    Alex Noble & Benjamin Salvesen

    captureTrainData.py:
        The purpose of this script is to generate training data that are used for training the keras model for use in classifyASL.py
        and classifyASL_RealTime.py. This script allows a user to generate 1000 images for each letter (a-e) that are the training and
        validation data used to train the keras model. An images folder is created that houses a subfolder for each letter, and the
        images for each letter are labeled "a1.png", "a2.png", and so on. The names of the subfolders are used by the model to discern classes,
        but the model does not use the actual file names for validation. The images themselves are altered in the sense that we calculate a 
        silhouette and contours on the hand to have the model learn about the shape of the letters in ASL more than the position/color/lighting, etc.

"""

# Imports for OpenCV computer vision, operating system access, and numpy for calculating silhouettes.
import cv2
import os
import numpy as np

"""
    getSilhouette: Extract and return the silhouettes/contours of the hand within the region of interest.

    Parameters:
    - frame: The current video frame to extract the hand silhouette from.
    - roiCoordinates: A tuple (x, y, roiSize) defining the top-left corner and size of the region of interest.

    Process:
    - Extract the ROI based on the provided coordinates.
    - Convert the ROI to grayscale.
    - Apply Gaussian blur to the grayscale image to reduce noise.
    - Use adaptive thresholding to binarize the image and make the contours easier to detect.
    - Find and fill the largest contour that is found (The hand against the background).
"""
def getSilhouette(frame, roiCoordinates):
    x, y, roiSize = roiCoordinates
    
    # Define the ROI with the roiCoordinates that are passed in
    roi = frame[y:y + roiSize, x:x + roiSize]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscaled ROI
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize the grayscaled, blurred ROI to make contours easier to detect
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    # Extract the silhouettes and contours from the ROI and set them to the silhouette frame
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    silhouetteFrame = np.zeros_like(roi, dtype=np.uint8)

    # Find the biggest contour and draw on them using red, everything else is black
    if contours:
        biggestContour = max(contours, key=cv2.contourArea)
        cv2.drawContours(silhouetteFrame, [biggestContour], -1, (0, 0, 255), thickness=cv2.FILLED)

    return silhouetteFrame

"""
    captureImages: Captures images of hand gestures for the various (a-e) ASL letters to use for training the model.
    Each letter's images are stored in separate subfolders within the main "images" folder.

    - ASL letters from 'a' to 'e' are captured.
    - Images are stored in a directory named 'images', with subfolders holding the images for each letter.
    - The function prompts the user to place their hand within a box on the screen and press/hold 'c' to capture.
    - Each image that is captured is processed to include only the silhouette and contours of the hand using the getSilhouette function.
"""
def captureImages():
    letters = ['a', 'b', 'c', 'd', 'e']
    
    # Set directory and check if it exists, if not, create it
    directory = 'images'
    os.makedirs(directory, exist_ok=True)
    
    # Define ROI size as 140 pixels
    roiSize = 140

    # Loop through each letter in the letters list and check that they have a directory/create a directory for them
    for letter in letters:
        letterDirectory = os.path.join(directory, letter)
        os.makedirs(letterDirectory, exist_ok=True)

        # Open the camera.
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Cannot open camera")
            exit()

        # Set image count to 0 and prompt user to take images for each respective letter
        count = 0
        print(f"Place your hand in the box to create training data for letter '{letter}'. Press/Hold 'c' to capture images.")
        
        # Loop to capture images of each ASL letter until 1000 images have been captured for each letter
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Error getting frame. Exiting.")
                break

            # Get the dimensions of the entire frame
            height, width, _ = frame.shape

            # Calculate the top left corner of the region of interest
            topLeft = ((width - roiSize) // 2, (height - roiSize) // 2)

            # Calculate the bottom right corner of the region of interest
            bottomRight = ((width + roiSize) // 2, (height + roiSize) // 2)

            # Store the x and y coordinates of the ROI's top left corner and the size of the ROI in a tuple
            roiCoordinates = (topLeft[0], topLeft[1], roiSize)

            # Process the ROI to get the hands silhouette/contours and overlay it back onto the frame
            processedROI = getSilhouette(frame, roiCoordinates)
            frame[topLeft[1]:topLeft[1] + roiSize, topLeft[0]:topLeft[0] + roiSize] = processedROI

            # Set up and show the frame with the ROI where the images are to be taken overlayed
            cv2.rectangle(frame, topLeft, bottomRight, (255, 0, 0), 2)
            cv2.imshow('frame', frame)
            
            # When "c" is pressed, capture an image and name it based on the respective letter and current count, and place it in its subfolder
            if cv2.waitKey(1) == ord('c'):
                imageName = f"{letter}{count+1}.png"
                imagePath = os.path.join(letterDirectory, imageName)

                # Save the processed ROI as an image in the respective letter's directory
                cv2.imwrite(imagePath, processedROI)
                count += 1
                print(f"Captured {imageName}")

                # Break once 1000 photos are taken for the current letter
                if count >= 1000:
                    break

        # Release the camera, destroy the cv2 windows, and output that images are done being taken for this letter.
        cam.release()
        cv2.destroyAllWindows()
        print(f"Completed capturing training images for letter '{letter}'.")

if __name__ == "__main__":
    captureImages()