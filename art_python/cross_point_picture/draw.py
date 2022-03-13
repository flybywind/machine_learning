# %%
import math
import numpy as np
import skimage as skim
import skimage.io as imo
from canvas import *

width = height = 1024
canvas = Canvas(width, height)
offset = 10
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
canvas.circle(center, 50, 500, Mode.Subtract)
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
