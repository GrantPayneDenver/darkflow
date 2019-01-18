# https://matplotlib.org/gallery/user_interfaces/embedding_in_wx5_sgskip.html

import wx
import os
import sys
import numpy as np
import re
import cv2
import copy
import wx.lib.agw.aui as aui
import wx.lib.scrolledpanel as scrolled

# where we save results
RES_DIR = r'C:\Users\grant\Documents\School\Deep Learning\Project_v2\trunk\training_boxes\dynamic_training_boxes\woman_and_two_dogs_scene'

# src dir, where we get the raw bounding boxes from
# SRC_DIR = r'C:\Users\grant\Documents\School\Deep Learning\Project_v2\trunk\training_boxes\all_boxes'
SRC_DIR = r'C:\Users\grant\Documents\School\Deep Learning\Project_v2\trunk\training_boxes\dogs_woman'

# file_addendum \\ helps make different scenes' file names be unique, skating scene doesn't have one
FILD_ADD = "_td" # for swing_set

# just do this for every 3rd frame. Analyzing changes over each frame is too fine grained and not worth it.
FRAME_INC = 1

# frame and image index presets, RESET AS NEEDED
IMG_IDX = 169
FRAME = 104


"""

TODO

For frame sets missing a match

Get all images from frame set
    concat the strings into a str
        if "match" not in str
            skip the frame set

"""

class Entity():
    # e = [image name, bitmap, [streams]]
    def __init__(self, image_name, wxBitMap, streams):
        self.image_name = image_name
        self.bitmap = wxBitMap
        self.cv2_imgs = [streams]

    def get_cv2_imgs(self):
        return self.cv2_imgs

    def append_to_cv2s(self, cv2_stream):
        self.cv2_imgs.append(cv2_stream)

    def update_bmp(self, bmp):
        self.bitmap.SetBitmap(bmp)

    def save_set(self, set_num, path=None):
        """ this saves the set of images in the current cv2 stack """
        try:
            if path:
                this_path = path
            else:
                this_path = self.working_dir = RES_DIR
            alpha = .6
            beta = 1 - alpha
            gamma = .5
            # blur new stream over current streams
            streams = self.get_cv2_imgs()
            base = streams[0]
            for e_str in streams[1:]:
                src1 = base
                src2 = e_str
                base = cv2.addWeighted(src1, alpha, src2, beta, gamma)
            # base = cv2.addWeighted(base, alpha, new_stream, beta, gamma)
            # save base image, set of stacked cv2 images
            cv2.imwrite(this_path+"\\%d_frame_%s_%s" % (set_num, FILD_ADD, self.image_name), base)
            pngBmpBytes = base.tobytes()
            bmp = wx.Bitmap.FromBuffer(250, 250, pngBmpBytes)
            if self.bitmap:
                self.update_bmp(bmp)
        except Exception as e:
            print(e)


class Data():

    def __init__(self):

        print("Grabbing Data")

        # folder = "./video_results/bounding_boxes"
        folder = SRC_DIR

        # images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        images = []
        for f in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, f)):
                images.append(f)

        def sorted_aphanumeric(data):
            print('sorted alphanumeric()')
            convert = lambda text: int(text) if text.isdigit() else text.lower()
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(data, key=alphanum_key)

        self.images_names = sorted_aphanumeric(images)

        raw_images = []
        grounds = []
        for i in range(0, len(images)):
            if "z" in images[i]:
                grounds.append(1)
            else:
                grounds.append(0)
            img = cv2.imread(folder + "/" + self.images_names[i], cv2.IMREAD_COLOR)
            raw_images.append(img)

        raw_images = np.array(raw_images)

        print(type(raw_images))
        print(type(raw_images[0]))
        print(raw_images.shape)
        print(raw_images[0].shape)

        self.raw_images = raw_images

    def GetImageData(self):
        return self.images_names, self.raw_images


class ImageProcFrame(wx.Frame):

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=u"Label Generation", pos=wx.DefaultPosition,
                          size=wx.Size(600, 400), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        # self.SetSizeHintsSz(wx.DefaultSize, wx.DefaultSize)
        d = Data()
        self.images_names, self.cv2_images = d.GetImageData()
        # self.image_dir = r'C:\Users\grant\Documents\School\Deep Learning\Project_v2\trunk\training_boxes\all_boxes'
        self.image_dir = SRC_DIR
        self.training_set_dir = r'C:\Users\grant\Documents\School\Deep Learning\Project_v2\trunk\training_boxes\dynamic_training_boxes\all_boxes_1'
        self.working_dir = RES_DIR

        # if not os.path.isdir(self.training_set_dir):
        #     os.makedirs(self.training_set_dir)

        # if not os.path.isdir(self.working_dir):
        #     os.makedirs(self.working_dir)

        # print(self.images)
        self.panels = []

        # iters thru the self.images files names
        # init at -1 for new sets
        # did 1 to 370 for skating_scene so far

        self.image_idx = IMG_IDX

        # tracks what sets are made per match set found
        # init at 0 for new sets
        # got frames 0 to 315 now for skating_scene
        self.set_frame = FRAME

        # tuple (imageFileName, image of StaticBitmap itself).
        # Holds all entities so far and their state so far.
        self.entities = []
        self.entities_idx = 0
        self.entitiesCopy = []

        self.CreateGUI()
        self.SetupEvents()

        self.ShowWarning()

    def ShowWarning(self):

        msg = wx.MessageDialog(self,
                               "Make sure you reset set_frame, image_idx, FILE_ADD, SRC_DIR and "
                               "RES_DIR for the current set you want", caption="Labeling GUI",
                      style=wx.OK | wx.ICON_WARNING, pos=wx.DefaultPosition)
        msg.ShowModal()
        msg.Destroy()

    def SetupEvents(self):
        """"""
        self.Bind(wx.EVT_CLOSE, self.Closing)
        self.addImageButton.Bind(wx.EVT_BUTTON, self.NextImage)
        self.prevImageButton.Bind(wx.EVT_BUTTON, self.PrevImage)
        self.Bind(wx.EVT_SIZE, self.Resizing)
        self.m_panel4.Bind(wx.EVT_SCROLLWIN_THUMBRELEASE, self.Scrolling)
        self.isNewButton.Bind(wx.EVT_BUTTON, self.NewEntity)
        self.classifyButton.Bind(wx.EVT_BUTTON, self.MatchFound)
        self.Bind(wx.EVT_KEY_DOWN, self.KeyPressed)
        # self.dontKnowButton.Bind(wx.EVT_BUTTON, self.DontKnow)
        # self.undoNewEntryButton.Bind(wx.EVT_BUTTON, self.UndoIsNew)

    def KeyPressed(self, event):
        keycode = event.GetUnicodeKey()
        print(keycode)

    def CreateGUI(self):
        """"""
        self.BodySizer = wx.BoxSizer(wx.VERTICAL)

        # entities panel, scrollable
        self.m_panel4 = wx.lib.scrolledpanel.ScrolledPanel(self, wx.ID_ANY, wx.DefaultPosition,
                                                           wx.DefaultSize, wx.TAB_TRAVERSAL)
        self.m_panel4.SetupScrolling()

        self.imageSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.m_panel4.SetSizer(self.imageSizer)
        self.m_panel4.Layout()
        self.imageSizer.Fit(self.m_panel4)
        self.BodySizer.Add(self.m_panel4, 2, wx.EXPAND | wx.ALL, 5)

        self.buttonsPanel = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        self.buttonsSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.buttonsPanel.SetSizer(self.buttonsSizer)
        # prev image button
        self.prevImageButton = wx.Button(self.buttonsPanel, wx.ID_ANY, "<<")
        self.buttonsSizer.Add(self.prevImageButton, 1, wx.SHRINK | wx.ALL, 5)
        # add image button
        self.addImageButton = wx.Button(self.buttonsPanel, wx.ID_ANY, ">>")
        self.buttonsSizer.Add(self.addImageButton, 1, wx.SHRINK | wx.ALL, 5)
        # classify button
        self.classifyButton = wx.Button(self.buttonsPanel, wx.ID_ANY, "Match")
        self.buttonsSizer.Add(self.classifyButton, 1, wx.SHRINK | wx.ALL, 5)
        # isNewButton button
        self.isNewButton = wx.Button(self.buttonsPanel, wx.ID_ANY, "Is New")
        self.buttonsSizer.Add(self.isNewButton, 1, wx.SHRINK | wx.ALL, 5)
        # # dontKnow button
        # self.dontKnowButton = wx.Button(self.buttonsPanel, wx.ID_ANY, "Dont Know")
        # self.buttonsSizer.Add(self.dontKnowButton, 1, wx.SHRINK | wx.ALL, 5)
        # # undo entry button
        # self.undoNewEntryButton = wx.Button(self.buttonsPanel, wx.ID_ANY, "Undo New")
        # self.buttonsSizer.Add(self.undoNewEntryButton, 1, wx.SHRINK | wx.ALL, 5)
        # text entry for label
        self.classifyTextCtrl = wx.TextCtrl(self.buttonsPanel)
        self.buttonsSizer.Add(self.classifyTextCtrl, 1, wx.SHRINK | wx.ALL, 5)
        self.buttonsPanel.Layout()
        # self.BodySizer.Add(self.controlsPanel, 1, wx.EXPAND | wx.ALL, 5)
        self.BodySizer.Add(self.buttonsPanel, 1, wx.EXPAND | wx.ALL, 5)

        # current image
        imageFile = r'C:\Users\grant\Documents\School\Deep Learning\Project_v2\trunk\test.png'
        self.currImagePanel = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        self.currImagePanelSizer = wx.BoxSizer(wx.HORIZONTAL)
        png = wx.Image(imageFile, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.currImageBitMap =  wx.StaticBitmap(self.currImagePanel, -1, png, (10, 5), (png.GetWidth(), png.GetHeight()))
        self.currImagePanelSizer.Add(self.currImageBitMap, 1, wx.EXPAND | wx.ALL, 5)
        self.currImagePanel.SetSizer(self.currImagePanelSizer)
        self.currImagePanel.Layout()
        self.currImagePanelSizer.Fit(self.currImagePanel)
        self.BodySizer.Add(self.currImagePanel, 2, wx.EXPAND | wx.ALL, 5)

        # current image label
        self.currLabel = wx.StaticText(self.m_panel4, wx.ID_ANY, label="-_-", size=(1, 15))
        self.BodySizer.Add(self.currLabel, 1, wx.EXPAND | wx.ALL, 5)

        self.SetSize((886, 812))
        self.SetMinSize((886, 812))
        self.SetSizer(self.BodySizer)
        self.Layout()
        self.Centre(wx.BOTH)



    '''
    In self.working_dir I need to store temps of all images in the scrolling pane so far
    With the new image blending over them. These need to be displayed, with their current 
    labels    
    '''
    """
    Save sets = {entities, curr image}
        entities of frame 1 entities = {e11, e12, e1n}
        curr image of frame1         = {c1_e11}, say if it matches e11

        curr image of frame 1 matches entity 1 of frame 1 
    """
    # todo, put the overlap of prev and nex image into this single
    def MovingImage(self):
        imageFile = self.images_names[self.image_idx]
        self.currLabel.SetLabel(imageFile)
        bmp = self.GetBitMapStream()
        # self.curr_cv2 = bmp
        self.currImageBitMap.SetBitmap(bmp)
        print('+- to :: %d %s' % (self.image_idx, imageFile))
        print('Current Frame: %d' % self.set_frame)
        self.classifyTextCtrl.SetValue(".jpg")
        self.classifyTextCtrl.SetFocus()
        self.classifyTextCtrl.SetInsertionPoint(0)
        bmp.Destroy()
        self.m_panel4.Layout()
        self.m_panel4.Refresh()
        self.Layout()


    def PrevImage(self, event):
        """ """
        self.image_idx -= FRAME_INC
        self.MovingImage()

    def NextImage(self, event):
        """ Dynamically update the entities in the entity scroller """
        self.image_idx += FRAME_INC
        self.MovingImage()

    def DontKnow(self, event):
        """ Broken """
        print("DontKnow()")
        self.set_frame += 1
        curr_stream = self.curr_cv2
        # save each entity stack
        for e in self.entities:
            e.save_set(self.set_frame)
        # save new image in working dir, use a temp entity
        # image_name, wxBitMap, streams
        stream = self.GetBitMapStream()
        bmp = wx.StaticBitmap(self, -1, stream, (10, 5), (stream.GetWidth(), stream.GetHeight()))
        # e = [image name, bitmap, [streams]]
        new_e = Entity("_UNKWN_", bmp, curr_stream)
        new_e.save_set(self.set_frame)
        del new_e

    def MatchFound(self, event):
        print("MatchFound()")
        curr_stream = self.curr_cv2
        ent_text_choice = self.classifyTextCtrl.GetValue()
        # save each entity stack
        for e in self.entities:
            if ent_text_choice != e.image_name:
                e.save_set(self.set_frame)

        # add new image to matched entity and update the bitmap stack
        for e in self.entities:
            if ent_text_choice == e.image_name:
                print('found match')
                e.append_to_cv2s(curr_stream)
                e.save_set(self.set_frame)
                # save new image in working dir, use a temp entity
                # image_name, wxBitMap, streams
                new_e = Entity("_match_%s" % e.image_name, None, curr_stream)
                new_e.save_set(self.set_frame)
                del new_e
        self.set_frame += 1

    def UpdateEntities(self, new_stream):
        """ Shows current stream over all entity streams"""
        alpha = .6
        beta = 1 - alpha
        gamma = .5
        self.entitiesCopy = copy.copy(self.entities)
        # e = [image name, bitmap, [streams]]
        for e_set in self.entities:
            # blur new stream over current streams
            streams = e_set.get_cv2_imgs()
            base = streams[0]
            for e_str in streams[1:]:
                src1 = base
                src2 = e_str
                base = cv2.addWeighted(src1, alpha, src2, beta, gamma)
            base = cv2.addWeighted(base, alpha, new_stream, beta, gamma)
            pngBmpBytes = base.tobytes()
            newBmp = wx.Bitmap.FromBuffer(250, 250, pngBmpBytes)
            e_set.update_bmp(newBmp)
            newBmp.Destroy()
            # png = wx.StaticBitmap(self, -1, stream, (10, 5), (stream.GetWidth(), stream.GetHeight()))
        self.Refresh()
        self.m_panel4.Layout()
        self.m_panel4.Refresh()

    def NewEntity(self, event):
        self.entities_idx += 1
        cv2_next = self.cv2_images[self.image_idx]
        stream = self.GetBitMapStream()
        bmp = wx.StaticBitmap(self, -1, stream, (10, 5), (stream.GetWidth(), stream.GetHeight()))
        # e = [image name, bitmap, [streams]]
        new_e = Entity(self.images_names[self.image_idx], bmp, cv2_next)
        self.entities.append(new_e)
        # self.entities.append([self.images_names[self.image_idx], png, [cv2_next]])
        imageFile = self.images_names[self.image_idx]
        label = wx.StaticText(self.m_panel4, wx.ID_ANY, label=imageFile, size=(1, 15))
        self.imageSizer.Add(bmp, 3, wx.ALL | wx.EXPAND, 5)
        self.imageSizer.Add(label, 1, wx.ALL | wx.SHRINK, 1)
        self.m_panel4.Layout()
        self.m_panel4.Refresh()

    def GetBitMapStream(self):
        """ grabs current cv2 image based on image_idx and converts to Bitmap"""
        cv2_next = self.curr_cv2 = self.cv2_images[self.image_idx]
        self.UpdateEntities(cv2_next)
        pngBmpBytes = cv2_next.tobytes()
        stream = wx.Bitmap.FromBuffer(250, 250, pngBmpBytes)
        return stream

    def Resizing(self, event):
        # print('resizing')
        self.Refresh()
        self.m_panel4.Layout()
        self.m_panel4.Refresh()
        event.Skip()

    def Scrolling(self, event):
        # EVT_SCROLLWIN_THUMBRELEASE
        self.Refresh()
        self.m_panel4.Layout()
        self.m_panel4.Refresh()
        event.Skip()

    def Closing(self, event):
        print('done')
        self.DestroyChildren()
        self.Destroy()
        sys.exit(0)

    def UndoIsNew(self, event):
        """ broken """
        lastEntryBmp, lastEntryLbl = self.entities[-1].bitmap, self.entities[-1].image_name
        imageFile = r'C:\Users\grant\Documents\School\Deep Learning\Project_v2\trunk\test.png'
        bmp = wx.Image(imageFile, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        lastEntryBmp.SetBitmap(bmp)
        self.entities.pop(-1)
        self.imageSizer.Clear()
        self.Refresh()
        self.m_panel4.Layout()
        self.m_panel4.Refresh()

        event.Skip()

def main():
    app = wx.App()
    appFrame = ImageProcFrame(parent=None)
    appFrame.Show()
    state = app.MainLoop()
    print(state)


if __name__ == "__main__":
    main()