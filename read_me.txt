
*
*	Object Tracking Project for Deep Learning
*

*
* by Debajit Kumar Sandilya and Grant Payne
*

*****      LINK TO CODE
*****
https://github.com/GrantPayneDenver/darkflow.git

              WEIGHTS

https://drive.google.com/file/d/1PsR-zlKV1CXZB-VzBSGwo1oZgVZQhDJe/view?usp=sharing
*****
*****

This code builds off of YOLO, but it really doesn't utilize YOLO for anything except
for bounding boxes.

We modified one file in YOLO, net/yolov2/predict.py

	This is where we added all of our bounding box pre-processing and saving code.
	
	We added tracker.py under net/yolov2/tracker.py
	
	This is where the trained CNN loads the trained weights and does object tracking 
	predictions.

	
1) to use this code, clone the repository listed above:


This code works in two parts. The YOLO "part" generates training data as well as 
prediction results in video form. Skip to part 2 of the directions for YOLO to video tracking predictions.

The training part exists in the Jupyter file called "trained_model_1".

1.1) to train

To train the model you need to set the boolean "train" to True in the Jupyter notebook file "tracking_model_1".

You will need to run the code and get to the part where it prompts "please enter a path to processed bounding box images for training or testing"

There you need to give it a path to the pre processed images from the YOLO side of the code.

These images should be the bounding box, blacked out images in: /training_boxes/all_boxes/

You can then train the model

1.2) saving weights

There will be an option to save the weights you generated. If you wish to skip, enter "0"


******
PART 2
TRACKING PART
******
2) make YOLO video with object tracked

To do this part, you need to clone our repo (if you haven't already) and run the YOLO code.

YOLO is run using the file called "flow", we converted it to a python file as flow.py
You can run it like so:

flow.py --model cfg/yolov2.cfg --load bin/yolov2.weights --demo dl_video.mp4 --saveVideo
The --model and --load parts are not important for us but are needed for YOLO to initialize.

This will start tracking the girl ice skating in the video called "dl_video.mp4", the --saveVideo flag will ensure
the tracking video results are saved as "video.mp4"
Please ensure dl_video.mp4 is in the same directory as flow.py.

2.1) as stated, this all runs in net/yolov2/predict.py and net/yolov2/tracker.py
 
in tracker.py, there will be a path that will want to grab the trained weights, you need to fix that path.

2.2) then in net/yolov2/predict.py set make_training_data = False (so now we want to use our model, not get training data :) )

Then you can run the code with "flow.py --model cfg/yolov2.cfg --load bin/yolov2.weights --demo dl_video.mp4 --saveVideo"


2.3) it will ask you to make a directory to store bounding boxes for training, enter s or S to skip this

2.4) after that YOLO should be on it's way pre-processing and allowing the model in tracker.py to predict. The video 
should come out as "video.mp4" in your working directory.

******



3) Running to do tracking predictions and getting just the images marked

3.1) to run the model use the same code
		set the boolean "train" to False
		
		
3.2) There will be a prompt to load weights
		specify the weights that can be found at this link: https://drive.google.com/file/d/1PsR-zlKV1CXZB-VzBSGwo1oZgVZQhDJe/view?usp=sharing
			(could not get these weights to go into github repo)

3.3 Make sure you paste in the full path to these weights after you download them onto your PC

3.4 There will be a prompt for a path to store the resulting images with predictions annotated 




