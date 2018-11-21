import numpy as np
import math
import cv2
import os
import json
import copy
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox

global frame
frame = 0

global out_dir
good_path = False
while not good_path:
	out_dir = input("Please enter a directory to store your output for bounding boxes")
	if os.path.isdir(out_dir): good_path = True

from ...cython_utils.cy_yolo2_findboxes import box_constructor

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""

	saved_boxes = []

	global frame
	frame += 1
	print('frame: ', frame)
	boxes = self.findboxes(net_out)

	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	img_num = 0
	resultsForJSON = []

	# for saving largest bounding box of girl
	max_box = None
	max_area = 0

	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		thick = int((h + w) // 300)
		if self.FLAGS.json:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue

		# save the image itself for each box
		image_copy = copy.copy(imgcv)
		# black out all of the image except for the box contained in left,top and right,bot
		black = (0,0,0)
		fill  = -1

		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		cv2.putText(imgcv, mess, (left, top - 12),
			0, 1e-3 * h, colors[max_indx],thick//3)

		cv2.rectangle(image_copy, (0,0), (w, top), black, fill)
		cv2.rectangle(image_copy, (0,top), (left, bot), black, fill)
		cv2.rectangle(image_copy, (0,bot), (w, h), black, fill)
		cv2.rectangle(image_copy, (right,top), (w, bot), black, fill)


		img_num += 1

		y = top
		x = left
		x2 = right
		y2 = bot
		roi = im[y:y2, x:x2]
		# saved_boxes.append([roi, image_copy])
		saved_boxes.append(image_copy)

		area = roi.shape[0] * roi.shape[1]

		if area > max_area:
			# max_box = [roi, image_copy]
			max_box = image_copy
			max_area = area

	global out_dir
	os.makedirs(out_dir+"\\frame%d" % frame)
	# cv2.imwrite((out_dir+"\\frame%d\\ground_box_frame%d.jpg" % (frame, frame)), max_box[0])
	# cv2.imwrite((out_dir+"\\frame%d\\ground_box_frame_black%d.jpg" % (frame, frame)), max_box[1])
	cv2.imwrite((out_dir+"\\frame%d\\ground_box_frame_black%d.jpg" % (frame, frame)), max_box)

	for i, bs in enumerate(saved_boxes):
		if bs is not max_box:
			# cv2.imwrite((out_dir + "\\frame%d\\box%d.jpg" % (frame, i)), bs[0])
			# cv2.imwrite((out_dir + "\\frame%d\\black_box%d.jpg" % (frame, i)), bs[1])
			cv2.imwrite((out_dir + "\\frame%d\\black_box%d.jpg" % (frame, i)), bs)

	if not save: return imgcv


	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(img_name, imgcv)
