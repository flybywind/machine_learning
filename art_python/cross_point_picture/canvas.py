from enum import Enum
from collections import namedtuple
import numpy as np
from skimage.draw import line, circle
import skimage.io as imo


Point = namedtuple('Point', ('X', 'Y'))

class Mode(Enum):
  OverWrite = 1
  Append = 2
  Subtract = 3

class Canvas():
  def __init__(self, width, height, dtype=np.uint8) -> None:
      self.width = width
      self.height = height
      self.dtype = dtype
      self.__canvas_data__ = np.zeros((height, width), dtype=dtype)
    
  def __safe_op(self, l, val, mode:Mode):
    if mode == Mode.Append:
      if self.dtype == np.uint8:
        self.__canvas_data__[l] = (self.__canvas_data__[l] +
                     np.minimum(255-self.__canvas_data__[l], val))
      else:
        self.__canvas_data__[l] += val 
    elif mode == Mode.OverWrite:
      self.__canvas_data__[l] = val 
    else:
      if self.dtype == np.uint8:
        self.__canvas_data__[l] = (self.__canvas_data__[l] -
                     np.minimum(self.__canvas_data__[l], val))
      else:
        self.__canvas_data__[l] -= val 

  def get_img(self):
    return self.__canvas_data__

  def line(self, loc_from:Point, loc_to:Point, val, mode:Mode = Mode.Append):
    l = line(loc_from.Y, loc_from.X, loc_to.Y, loc_to.X)
    self.__safe_op(l, val, mode)

  def circle(self, center:Point, val, radius, mode:Mode = Mode.Append):
    c = circle(center.Y, center.X, radius, shape=(self.height, self.width))
    self.__safe_op(c, val, mode)

  def find_mark(self, val, mark=False):
    mark_point = np.argwhere(self.__canvas_data__ > val)
    if mark:
      mark_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
      for p in mark_point:
        mark_img[p[0], p[1], :] = [255, 0, 0]
      return mark_point, mark_img
    return mark_point, None

  def show(self):
    ''' only useful for interactive debug
    '''
    imo.imshow(self.__canvas_data__)