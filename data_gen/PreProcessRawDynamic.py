import cv2
import os
import re

SET_MOD = "walk"
DST = r'C:\Users\grant\Documents\School\Deep Learning\Project_v2\trunk\training_boxes\dynamic_training_boxes\FINALIZED'
SRC = r'C:\Users\grant\Documents\School\Deep Learning\Project_v2\trunk\training_boxes\dynamic_training_boxes\woman_and_two_dogs_scene'
DEMO = False # mark the cv2 writing with something visual to indicate it's getting the y or n designation right


class Image():
    def __init__(self, imgPath):
        self.imgPath = imgPath
        self.cv2 = cv2.imread(imgPath)
        self.name = imgPath.split("\\")[-1]
        self.frm_num = str(self.name.split("_")[0])
        self.end_num = str(self.name.split("_")[-1]).strip(".jpg")
        self.the_cover = "match" in self.name
        pass

def main():

    counter = 0
    set_dict = {} # [int] -> [Image, Image,..,Image] all images in a frame

    folder = SRC
    images_names = []
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, f)):
            images_names.append(f)

    def sorted_aphanumeric(data):
        print('sorted alphanumeric()')
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)

    images_names = sorted_aphanumeric(images_names)
    cv2_images = []
    for index, img_name in enumerate(images_names):
        img = Image(folder + "\\" + img_name)
        cv2_images.append(img)
        if str(index) not in set_dict.keys():
            set_dict[str(index)] = []
        # str(index) in set_dict.keys():
        set_dict[img.frm_num].append(img)

    for frame_num, frame_images in set_dict.items():
        if not frame_images: continue
        the_cover = None
        for fi in frame_images:
            if fi.the_cover:
                if the_cover is not None:
                    break # should never find the_cover twice in one set
                the_cover = fi
                break
        if the_cover:
            for fi in frame_images:
                if fi is the_cover: continue
                overlayCoverOnFI(the_cover, fi, the_cover.end_num == fi.end_num, counter)

        counter += 1

def overlayCoverOnFI(the_cover, fi, is_match_set, counter):
    alpha = .6
    beta = 1 - alpha
    gamma = .5
    # blur cover cv2 over fi cv2
    overlayed_set = cv2.addWeighted(fi.cv2, alpha, the_cover.cv2, beta, gamma)
    if is_match_set:
        indicator = "y"
    else:
        indicator = "n"
    if DEMO:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlayed_set, str(indicator), (10, 60), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(DST + "\\%d_%s_%s.jpg" % (counter, SET_MOD, indicator), overlayed_set)


if __name__ == "__main__":
    main()

