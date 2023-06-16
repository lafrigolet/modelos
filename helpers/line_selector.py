#!/usr/bin/env python3
# coding: utf-8

# In[1]:


from cmath import inf
from os import stat
from turtle import left, right
from typing import Tuple, List
import math
from cv2 import CHAIN_APPROX_NONE
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
from scipy.spatial import distance
import statistics


# In[2]:


def mm2pix(mm:int, dpi: int):
    return mm * dpi // 254

class Word:
    
    def __init__(self, left : int, top : int , right : int , bottom : int,  image : np.ndarray):
        self._left   = left
        self._top    = top
        self._right  = right
        self._bottom = bottom
        self._image  = image
        self._mid    = None
        self._image_mean = None


    def get_height(self) -> int:
        return 1 + self._bottom - self._top


    def get_width(self) -> int:
        return 1 + self._right - self._left


    def get_left(self) -> int:
        return self._left


    def get_top(self) -> int:
        return self._top


    def get_right(self) -> int:
        return self._right


    def get_bottom(self) -> int:
        return self._bottom


    def get_image(self) -> np.ndarray:
        return self._image


    def get_tuple(self) -> Tuple[int,int,int,int]:
        return (self._left,self._top,self._right,self._bottom)


    def __eq__(self,other) -> bool:
            return  self.get_tuple() == other.get_tuple()


    def __lt__(self,other) -> bool:
        if self._left != other._left : 
            return self._left < other._left
        elif self._top != other._top : 
            return self._top < other._top
        elif self._right != other._right : 
            return self._right < other._right
        else : 
            return self._bottom < other._bottom


    def get_image_mean(self) -> Tuple[int,int]:
        if self._image_mean is None:
            image = self._image[self._top:self._bottom+1,self._left:self._right+1]
            image = image / 255

            column_sums = np.sum(image,0)
            row_sums = np.sum(image,1)

            columns = image.shape[1] + 1
            rows    = image.shape[0] + 1

            column_multipliers = np.array(list(range(0,columns-1)))
            row_multipliers    = np.array(list(range(0,rows-1)))

            total = np.sum(column_sums)

            x = np.sum(column_sums * column_multipliers) / total
            y = np.sum(row_sums    * row_multipliers)    / total

            self._image_mean = (self._left + x, self._top + y)
        
        return self._image_mean


    def join(self, word : 'Word'):
        self._image_mean = None
        self._left   = min(self._left, word._left)
        self._top    = min(self._top, word._top)
        self._right  = max(self._right, word._right)
        self._bottom = max(self._bottom, word._bottom)
        self._mid    = None
        self._image_mean = None


    def get_geomteric_mean(self) -> Tuple[int,int]:
        return ((self._top + self._bottom) // 2,(self.left + self._right) // 2)


    def _calculate_mid(self):
        if self.get_height() > self.get_width():
            self._mid = (self._top + self._bottom) // 2
        elif self._mid is None:
            image = self._image[self._top:self._bottom+1,self._left:self._right+1]
            histo = np.sum(image,1)
            self._mid = self._top + np.argmax(histo)


    def get_start(self) -> Tuple[int,int]:
        if self._mid == None:
            self._calculate_mid()

        return (self._left, self._mid)


    def get_end(self) -> Tuple[int,int]:
        if self._mid == None:
            self._calculate_mid()

        return (self._right, self._mid)


# In[3]:


class Line():


    def __init__(self, word : Word, word_join_threshold : int = 0):
        self._words = [word]
        self._left = word.get_left()
        self._top = word.get_top()
        self._right = word.get_right()
        self._bottom = word.get_bottom()
        self._word_join_threshold = word_join_threshold


    def distance(self, word : Word) -> float:     
        last_word = self._words[-1] 
        last_end = last_word.get_end()
        last_image_mean = last_word.get_image_mean()

        next_start = word.get_start()
        next_image_mean = word.get_image_mean()

        vertical_distance = abs(last_image_mean[1] - next_image_mean[1])

        if vertical_distance > 1.5 * max(last_word.get_height(), word.get_height()):
            return np.inf

        return distance.euclidean(last_end,next_start) + abs(last_end[1]-next_start[1])


    def add(self, word : Word):
        last_word = self._words[-1]
        if abs(word._right - last_word._right) < self._word_join_threshold:
            last_word.join(word)
            word = last_word
        else:
            self._words.append(word)

        if word.get_left() < self._left:
            self._left = word.get_left()
        if word.get_top() < self._top:
            self._top = word.get_top()
        if word.get_bottom() > self._bottom:
            self._bottom = word.get_bottom()
        if word.get_right() > self._right:
            self._right = word.get_right()


    def get_base_line(self) -> List[Tuple[int,int]]:
        base_line = []
        for word in self._words:
            base_line.append(word.get_start())
            base_line.append(word.get_end())
        return base_line


    def get_words(self) -> List[Word]:
        return self._words

    def get_width(self) -> int:
        return self._right - self._left + 1

    def get_height(self) -> int:
        return self._bottom - self._top + 1

    def get_top(self) -> int:
        return self._top

    def get_bottom(self) -> int:
        return self._bottom

    def get_left(self) -> int:
        return self._left

    def get_right(self) -> int:
        return self._right


# In[4]:


def _word_analysis(word_list : List[Word]) -> List[Word]:

    def _seems_bimodal(word : Word) -> int :
        histo = np.sum(word.get_image(),1)
        len_histo = histo.shape[0]
        p1 = int(len_histo * 0.4)
        p2 = int(len_histo*0.6)
        z1 = np.sum(histo[0:p1])
        z2 = np.sum(histo[p1:p2])
        z3 = np.sum(histo[p2:])

        if (z1 > 2*z2) and (2*z2 < z3):
            return (p2 + p1) // 2
        else:
            return 0

    def _compute_height_mean(word_list):
        height_list = [word.get_height() for word in word_list]
        height_mean = statistics.mean(height_list)
        height_deviation = statistics.stdev(height_list)
        return height_mean,height_deviation

    height_mean, height_deviation = _compute_height_mean(word_list)

    new_word_list = []
    for word in word_list:
        if (word.get_height() > height_mean + height_deviation):
            cut_point = _seems_bimodal(word)
            if cut_point != 0:
                _seems_bimodal(word)
                word_top = word.get_image()[0:cut_point,:]
                word_bottom = word.get_image()[cut_point:,:]
                new_word_list.append(Word(word.get_left(), word.get_top(), word_top))
                new_word_list.append(Word(word.get_left(), word.get_top() + word_top.shape[0], word_bottom))
            else:
                new_word_list.append(word)
        else:
            new_word_list.append(word)

    return new_word_list


def _locate_words(
    bin_img : np.ndarray, 
    contours : List[Tuple], 
    min_word_dimmension : int) -> List[Word]:
    """_summary_
    Args:
        bin_img (np.ndarray): Binary image in which 0 is used for background and 255 for ink.
        contours (List[Tuple]): 
        min_word_dimmension (int, optional): minimun dimmesion of a words to be considered as a word, in milimeters.
    Returns:
        _type_: List[Word]
    """

    word_list = []

    for contour in contours[0]:
        left = np.min(contour[:,0,0])
        right = np.max(contour[:,0,0])
        top = np.min(contour[:,0,1])
        bottom = np.max(contour[:,0,1])

        if (1 + bottom - top) >= min_word_dimmension and             (1 + right - left) >= min_word_dimmension and             (1 + bottom - top) < 1 + min_word_dimmension * 10:

            word = Word(left, top, right, bottom, bin_img)
            word_list.append(word)
    return word_list


def _binarize_image(img: np.ndarray):
    """Binarize a image and set 0 for background and 255 for ink.
    Args:
        img (np.ndarray): gray image of a handwritten document.
    """
    threshold, bin_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = 255 - bin_img
    return bin_img


# In[5]:


class Page:
    
    def __init__(self, 
        img: np.ndarray,
        separation_threshold:int,
        min_word_dimmension : int,
        break_connected_lines : bool,
        word_join_threshold : int):

        """Create a page model from a list of words.
        Args:
            img (np.ndarray): gray image of a handwritten document.
            separation_threshold (int): maximun separation between words or letters in a line, in milimeters.
            min_word_dimmension (int): minimun dimmesion of a words to be considered as a word, in milimeters.
            break_connected_lines (bool) : 
        """
        self._original_img     = img
        self._bin_img = _binarize_image(img)
        contours = cv2.findContours(self._bin_img, mode = cv2.RETR_LIST, method = CHAIN_APPROX_NONE)

        word_list = _locate_words(self._bin_img, contours, min_word_dimmension)
        self._lines = []

        if word_list == []:
            return

        if break_connected_lines:
            word_list = _word_analysis(word_list)

        word_list.sort()

        for word in word_list:

            min_distance = inf
            choosen_line : Line = None 
            for line in self._lines:
                distance = line.distance(word)

                if (distance <= separation_threshold) and (distance < min_distance):
                    min_distance = distance
                    choosen_line = line

            if choosen_line is not None:
                choosen_line.add(word)
            else:
                self._lines.append(Line(word,word_join_threshold))

    def get_lines(self) -> List[Line]:
        return self._lines

    def get_image(self):
        return self._bin_img
    
    def get_original_img(self):
        return self._original_img


# In[6]:


def handwritten_text_line_detection(
    img : np.ndarray, 
    separation_threshold : int = 127 ,
    min_word_dimmension : int = 20,
    break_connected_lines : bool = True,
    dilate_ink : bool = False,
    word_join_threshold : int = 0,
    dpi : int = 200) -> Page:
    
    """Detects horizontal handwritten text lines in a gray image.
    Args:
        img (np.ndarray): gray image of a handwritten document.
        separation_threshold (int, optional): maximun separation between words or letters in a line, in milimeters. Defaults to 127 mm.
        min_word_dimmension (int, optional): minimun dimmesion of a words to be considered as a word, in milimeters. Defaults to 20 mm.
        break_connected_lines (bool, optional) : try to break lines of text if they are connected in some point. Default true.
        word_join_threshold (int, optional): join words nearer than this threshold. Default to 0 mm.
        dpi (int, optional): resolution of the image in dots per inch. Defaults to 200 DPI.
    Returns:
        Page: Object that contains the lines and words coordinates.
    """
    assert img.dtype == np.uint8, "The image must be an uint8 numpy array"
    assert len(img.shape) == 2, "The image must be 2D numpy array"
    assert sum(sum(img.astype(int))) > 128*img.shape[0]*img.shape[1], "The image must be mostly white (255)"

    if dilate_ink:
        img = cv2.erode(img,np.ones((3,3)))
        
    #plt.imshow(img, cmap = "gray")
    #plt.show()
    
    page = Page(img, 
                separation_threshold = mm2pix(separation_threshold,dpi),
                min_word_dimmension = mm2pix(min_word_dimmension,dpi), 
                break_connected_lines = break_connected_lines,
                word_join_threshold = mm2pix(word_join_threshold,dpi))
    
    #plt.imshow(page.get_image(), cmap = "gray")
    #plt.show()

    return page


# In[7]:


def draw_page(page:Page, show_lines = True, show_base_lines = True, show_words = True):
    plt.figure(figsize = (20,20))
    extent = (0, page.get_image().shape[1], page.get_image().shape[0], 0)
    plt.imshow(255-page.get_image(), cmap = "gray", extent = extent)
    ax = plt.gca()

    for line in page.get_lines():

        if show_lines:
            line_box = Rectangle((line.get_left(),line.get_top()),line.get_width(),line.get_height(),color="blue",fill=False,linewidth=1)
            ax.add_patch(line_box)

        if show_words:
            for word in line.get_words():
                height = word.get_height()
                width = word.get_width()
                xy = (word.get_left(),word.get_top())
                rect = Rectangle(xy,width,height,color="green",fill=False,linewidth=1)
                ax.add_patch(rect)

        if show_base_lines:
            print(line.get_base_line())
            line_poly = Polygon(line.get_base_line(),color="red",fill=False,closed=False,linewidth=1)
            ax.add_patch(line_poly)

    plt.show()   


# In[8]:


def test_get_real_4():
    print("Test get real 4")
    img = cv2.imread("dataset/source/176-1.png",0)
    page = handwritten_text_line_detection(img,dpi=300,break_connected_lines=False,dilate_ink=True,min_word_dimmension=10)
    draw_page(page, show_words= False, show_lines=False)
    #draw_page(page, show_words= False)
    #draw_page(page)


# In[9]:


# test_get_real_4()


# In[ ]:


def crop_page(page:Page, h = 40, w = 40):
    img    = page.get_original_img()
    extent = (0, page.get_image().shape[1], page.get_image().shape[0], 0)
    cropped_images = []

    #plt.figure(figsize = (20,20))
    #plt.imshow(255-page.get_image(), cmap = "gray", extent = extent)
    #plt.show()   
    
    for line in page.get_lines():
        # print(line.get_base_line())
        for dot in line.get_base_line():
            x, y = dot
            #print(max(0, y-h//2), min(y+h//2, page.get_image().shape[1]) , max(0,x-w//2), min(x+w//2, page.get_image().shape[0]))
            cropped_image = img[max(0, y-h//2):min(y+h//2,page.get_image().shape[0]) , max(0, x-w//2):min(x+w//2, page.get_image().shape[1])]
            if (cropped_image.shape[0] == h and cropped_image.shape[1] == w): # only images with the right size are added to the cropped_images
                cropped_images.append(cropped_image)
        
    return cropped_images


def crop_page_to_dir(input_image_path, outputdir = '.', h=30, w=150):
    img = cv2.imread(input_image_path, 0)
    page = handwritten_text_line_detection(img,dpi=300,break_connected_lines=False,dilate_ink=True,min_word_dimmension=10)
    cropped_images = crop_page(page, h, w)
    
    for i, cropped_image in enumerate(cropped_images):
        filepath = outputdir + '/cropped_image_' + str(i) + '.jpg'
        print(filepath)
        #plt.imshow(cropped_image, cmap = "gray")
        #plt.show()
        cv2.imwrite(filepath, cropped_image)

#crop_page_to_dir('dataset/source/176-1.png', 'dataset/source/0')
#crop_page_to_dir('dataset/source/224-1.png', 'dataset/source/1')

