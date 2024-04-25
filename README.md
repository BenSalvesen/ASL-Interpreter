ASL Interpreter
Alex Noble & Benjamin Salevsen


Before Running:
- Ensure that all files are present:
	1. "images" - Folder containing 5 subfolders, one for each letter
		"a" - Subfolder containing 1000 images of "a" in ASL
		"b" - Subfolder containing 1000 images of "b" in ASL
		"c" - Subfolder containing 1000 images of "c" in ASL
		"d" - Subfolder containing 1000 images of "d" in ASL
		"e" - Subfolder containing 1000 images of "e" in ASL

	IMAGE FOLDER TOO BIG TO PUT ON GITHUB, TRAINING CAN BE DONE OR WATCH THE DEMO HERE
	========LINK COMING SOON=======	
  
	3. captureTrainData.py
		Script that allows a user to capture images for the above subclasses/letters to be classified in real time.
	4. trainModel.py
		Script that loads the training images, trains, and saves the model
	5. classifyASL_RealTime
		Script that opens the camera and displays a box that is the region of interest, allowing a user to place their hand in the a, b, c, d, or e configuration and classifies what it sees in the window every second.

	6. README (this document):
		Outlines how to run the project files
	7. ASLInterpreterReport.docx
		Report on the project and methods used.

- Ensure that Python 3.11.9 is installed.
	After installing Python 3.11.9, run the following commands in VSCode's python terminal:
	- pip install tensorflow
	- pip install keras
	- pip install opencv-python
	- pip install numpy

Note: tensorflow imports will display as unresolved. This comes down to a Pylance misconfiguration, but the code will still run as follows, even with the warnings.


To Run:
Gathering Data (If image files are not able to be compressed and submitted to canvas):
	- Run the "captureTrainData.py" script:
		- You will be prompted to hold your hand in the box in the middle and press "c" to capture images for each respective letter. Hold your hand toward the base of the box to ensure that your wrist is not getting captured, which would skew classification.
		- Hold down the "c" key and hold your hand still, slightly turning it to the left and right to capture the silhouettes and contours as best as possible.
		- Repeat for all 5 letters, and the images will be stored in the "images" folder, under a subfolder for each letter respectively.
	
	- Run the "trainModel.py" script:
		- The model will begin training, going through epochs using the 5000 image files for training and validation (not mixed together)
		- After training is complete, the model will be saved to the directory and titled "classifyASL.keras"
	
	- Run the "classifyASL_RealTime.py" script:
		- The camera will come up and a prompt will say "waiting for gesture"
		- Place your hand in the box like when creating the data, and a prediction will be made every second.
		- Press q to quit/terminate the program.
