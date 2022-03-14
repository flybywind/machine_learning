# %%
from collections import namedtuple
import math
import numpy as np
import skimage as skim
import skimage.io as imo
from canvas import *

width = height = 1024
canvas = Canvas(width, height)
offset = 3
center = Point(canvas.width//2, canvas.height//2)
center1 = Point(canvas.width//2-offset, canvas.height//2)
center2 = Point(canvas.width//2+offset, canvas.height//2)
center3 = Point(canvas.width//2, canvas.height//2-offset)
center4 = Point(canvas.width//2, canvas.height//2+offset)
canvas.circle(center, 50, 505)
canvas.circle(center1, 50, 505, Mode.OverWrite)
canvas.circle(center2, 50, 505, Mode.OverWrite)
canvas.circle(center3, 50, 505, Mode.OverWrite)
canvas.circle(center4, 50, 505, Mode.OverWrite)
canvas.circle(center, 50, 503, Mode.Subtract)
canvas.show()
# %%
degrees = np.linspace(0, np.pi, 90)
vertical_center = (Point(center.X, 0), Point(center.X, height-1))

for i, rad in enumerate(degrees):
  if rad <= np.pi/4:
    loc_from =  Point(
      round(center.X + math.tan(rad) * center.Y), 0)
    loc_to =  Point(
      round(center.X - math.tan(rad) * center.Y), height-1)
  elif rad <= 3*np.pi/4:
    loc_from =  Point(width-1, 
      round(center.Y - center.X/math.tan(rad)))
    loc_to =  Point(0,
      round(center.Y + center.X/math.tan(rad)))
  else:
    loc_from =  Point(round(center.X + math.tan(np.pi - rad) * center.Y), height-1)
    loc_to =  Point(round(center.X - math.tan(np.pi - rad) * center.Y), 0)

  loc_from = Point(np.clip(loc_from.X, 0, width-1),
                   np.clip(loc_from.Y, 0, height-1))
  loc_to = Point(np.clip(loc_to.X, 0, width-1),
                 np.clip(loc_to.Y, 0, height-1))
  canvas.line(loc_from, loc_to, 200)
  # print(f"draw line at {i}th rad {rad}: {loc_from} --> {loc_to}")

canvas.circle(center, 200, 490, Mode.Subtract)
canvas.show()
# %%
mark_point, img = canvas.find_mark(210, True)
imo.imshow(img)
# %%
RadLoc = namedtuple("RadLoc", ["X", "Y", "Deg", "Delta"])
whole_circle_deg = np.concatenate((degrees,
     degrees[1:-1] + np.pi))
# can't init a list of list in this way, the list item share the same reference!
mark_point_map = {}
for p in mark_point:
  x, y = p[1] - center.X, center.Y - p[0]
  deg = 0.0
  if x > 0 and y >= 0:
    # quatrant 1
    if y == 0:
      deg = np.pi/2
    else:
      deg = math.atan(x/y)
  elif x > 0 and y < 0:
    # quatrant 2
    deg = np.pi/2 + math.atan(-y/x)
  elif x <= 0 and y > 0:
    # quatrant 4:
    if x == 0:
      deg = 0.0
    else:
      deg = np.pi*2 - math.atan(-x/y)
  elif x <= 0 and y <= 0:
    # quatrant 3:
    if y == 0:
      deg = 3*np.pi/2
    else:
      deg = np.pi + math.atan(x/y)
  else:
    raise RuntimeError(f"invalid location: ({x}, {y})")
  ind = np.argmin(np.abs(whole_circle_deg - deg))
  diff_val = math.fabs(whole_circle_deg[ind] - deg)
  cur_loc = mark_point_map.get(ind, RadLoc(0, 0, 0, np.inf))
  if cur_loc.Delta > diff_val:
    mark_point_map[ind] = RadLoc(X=p[1], Y=p[0], 
        Deg=whole_circle_deg[ind], Delta=diff_val)

# %%
