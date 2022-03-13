# %%
import numpy as np
import skimage as skim
import skimage.io as imo
from canvas import *


canvas = Canvas(1024, 1024)
center = Point(canvas.width//2, canvas.height//2)
canvas.circle(center, 50, 504)
canvas.circle(center, 50, 500, Mode.Subtract)
# canvas.show()

