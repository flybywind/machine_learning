# %%
from collections import namedtuple
import math
from datetime import datetime,timedelta
import os
import numpy as np
import skimage as skim
import skimage.io as imo
from skimage.color import rgb2gray
from skimage.transform import rescale
from tensorforce.agents import Agent
from drawing_env import DrawingEnvironment
from os import path

from canvas import *
import shutil

def del_all_in_path(basePath):
  del_files = os.listdir(basePath)
  del_files = [path.join(basePath, f) for f in del_files]
  for f in del_files:
    if path.isdir(f):
      continue
    open(f, 'w').close()
    os.remove(f)

def save_ckp(agent, dir):
  try:
    del_all_in_path(dir)
    shutil.rmtree(dir)
  except:
    pass
  try:
    os.mkdir(dir)
    print(f"\n------------\nstart saving model ... \n")
    agent.save(directory=dir,
             format='checkpoint', append='timesteps')
  except Exception as e:
    print(f"[Exception] save checkpoint exception: {e}")

width = height = 255
radius = width // 2 - 20
canvas = Canvas(width, height)
offset = 2
center = Point(canvas.width // 2, canvas.height // 2)
center1 = Point(canvas.width // 2 - offset, canvas.height // 2)
center2 = Point(canvas.width // 2 + offset, canvas.height // 2)
center3 = Point(canvas.width // 2, canvas.height // 2 - offset)
center4 = Point(canvas.width // 2, canvas.height // 2 + offset)
canvas.circle(center, 50, radius + 3)
canvas.circle(center1, 50, radius + 3, Mode.OverWrite)
canvas.circle(center2, 50, radius + 3, Mode.OverWrite)
canvas.circle(center3, 50, radius + 3, Mode.OverWrite)
canvas.circle(center4, 50, radius + 3, Mode.OverWrite)
canvas.circle(center, 50, radius, Mode.Subtract)
# canvas.show()
# %%
degrees = np.linspace(0, np.pi, 90)

for i, rad in enumerate(degrees):
    if rad <= np.pi / 4:
        loc_from = Point(
            round(center.X + math.tan(rad) * center.Y), 0)
        loc_to = Point(
            round(center.X - math.tan(rad) * center.Y), height - 1)
    elif rad <= 3 * np.pi / 4:
        loc_from = Point(width - 1,
                         round(center.Y - center.X / math.tan(rad)))
        loc_to = Point(0,
                       round(center.Y + center.X / math.tan(rad)))
    else:
        loc_from = Point(round(center.X + math.tan(np.pi - rad) * center.Y), height - 1)
        loc_to = Point(round(center.X - math.tan(np.pi - rad) * center.Y), 0)

    loc_from = Point(np.clip(loc_from.X, 0, width - 1),
                     np.clip(loc_from.Y, 0, height - 1))
    loc_to = Point(np.clip(loc_to.X, 0, width - 1),
                   np.clip(loc_to.Y, 0, height - 1))
    canvas.line(loc_from, loc_to, 200)
    # print(f"draw line at {i}th rad {rad}: {loc_from} --> {loc_to}")

canvas.circle(center, 255, radius - 10, Mode.Subtract)
# canvas.show()
# %%
mark_point, img = canvas.find_mark(210, True)
# imo.imshow(img.astype(np.uint8))
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
            deg = np.pi / 2
        else:
            deg = math.atan(x / y)
    elif x > 0 and y < 0:
        # quatrant 2
        deg = np.pi / 2 + math.atan(-y / x)
    elif x <= 0 and y > 0:
        # quatrant 4:
        if x == 0:
            deg = 0.0
        else:
            deg = np.pi * 2 - math.atan(-x / y)
    elif x <= 0 and y <= 0:
        # quatrant 3:
        if y == 0:
            deg = 3 * np.pi / 2
        else:
            deg = np.pi + math.atan(x / y)
    else:
        raise RuntimeError(f"invalid location: ({x}, {y})")
    ind = np.argmin(np.abs(whole_circle_deg - deg))
    diff_val = math.fabs(whole_circle_deg[ind] - deg)
    cur_loc = mark_point_map.get(ind, RadLoc(0, 0, 0, np.inf))
    if cur_loc.Delta > diff_val:
        mark_point_map[ind] = RadLoc(X=p[1], Y=p[0],
                                     Deg=whole_circle_deg[ind], Delta=diff_val)
# %%

anchor_points = [Point(v.X, v.Y) for _, v in mark_point_map.items()]
print(f"mark points: {anchor_points[:10]}[len={len(anchor_points)}]")
ref_img = imo.imread("image/v-for-vendetta-mask.png")
# ref_img = rgb2gray(ref_img)
ref_img = rescale(rgb2gray(ref_img[:, :, :3]), 0.2)
# h = 271, w = 258
ref_img2 = np.zeros((height, width), dtype=np.float64)
h, w = ref_img.shape
h_offset = (height - h) // 2
w_offset = (width - w) // 2

# move the face to the center
h_shift = 15
h_offset2 = h_offset - h_shift
ref_img2[h_offset2:h_offset2 + h, w_offset:w_offset + w] = ref_img
ref_img2[:h_offset2, :] = 1
ref_img2[-h_offset-h_shift:, :] = 1
ref_img2[:, :w_offset] = 1
ref_img2[:, -w_offset-10:] = 1
ref_img2 = np.clip(1-ref_img2, 0, 1)
# imo.imshow(ref_img2)
# imo.show() # show() can block execution
# %%


basePath = "."
environment = DrawingEnvironment(ref_img2, 0.02,
                    anchor_points,
                    max_time_stamp=12000,
                    contrast=0.1)

start = datetime.now()

print(f"{start}: init env and agent ...")
ckp_path = "checkpoint"
load_suc = False
if path.isdir(ckp_path):
  try:
    agent = Agent.load(ckp_path, format='checkpoint', environment=environment)
    load_suc = True
    print(f"load agent from checkpoint success!")
  except Exception as e:
    print(f"Error: try loading checkpoint failed, {e}")

if not load_suc:
  agent = Agent.create(agent=path.join(basePath, 'drawer_tensorforce.json'),
                     environment=environment,
                     config={
                         'device': 'GPU'
                     })
print(f"agent network: {agent.get_architecture()}")

cpk_path = path.join(basePath, "checkpoint")
num_updates = 0
for episode in range(10000):
    # Episode using act and observe
    states = environment.reset()
    terminal = False
    sum_rewards = 0.0
    print(f"{datetime.now()}: start Episode {episode} ...")
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        num_updates += agent.observe(terminal=terminal, reward=reward)
        sum_rewards += reward
        if datetime.now() - start > timedelta(seconds=0.5*60*60):
          save_ckp(agent, cpk_path)
          start = datetime.now()
    print('Episode {}: return={} updates={}'.format(episode, sum_rewards, num_updates))
    save_ckp(agent, cpk_path)

# Evaluate for 100 episodes
sum_rewards = 0.0
for _ in range(100):
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(
            states=states, internals=internals, independent=True, deterministic=True
        )
        states, terminal, reward = environment.execute(actions=actions)
        sum_rewards += reward
print('Mean evaluation return:', sum_rewards / 100.0)
# Close agent and environment
agent.close()
environment.close()
# %%
