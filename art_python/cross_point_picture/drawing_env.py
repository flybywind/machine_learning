###-----------------------------------------------------------------------------
##
from collections import deque
from canvas import Canvas, Point, Mode
from datetime import datetime
import math
import numpy as np
from os import path
from tensorforce.environments import Environment


def gauss_mask(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function
  """
  x, y = np.mgrid[-size[0] // 2 + 1:size[0] // 2 + 1, -size[1] // 2 + 1:size[1] // 2 + 1]
  g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
  return g


###-----------------------------------------------------------------------------
### Environment definition
class DrawingEnvironment(Environment):
  """This class defines a simple drawing environment.
  The award is the image diff value between current image the reference image
  ref_img: float matrix, gray value in [0, 1]
  """

  def __init__(self, ref_image, delt_val: float, anchor_points: list,
               contrast: float = 0.2,
               max_time_stamp: int = -1,
               action_fifo_len: int = 15,
               basePath="."):
    self.ref_image = ref_image.astype(np.float)
    self.img_shape = ref_image.shape
    self.delt_val = delt_val
    self.contrast = contrast

    self.mask_img = (self.ref_image > self.contrast) + 0.0
    self.mask_img_neg = 1 - self.mask_img
    mask_sum = np.sum(self.mask_img)
    mask_sum_neg = ref_image.size - mask_sum
    self.neg_discount = mask_sum / mask_sum_neg
    self.gauss_weight = gauss_mask(self.img_shape, np.max(self.img_shape) * 0.5)
    print(f"mask sum pos = {mask_sum}, neg = {mask_sum_neg}")
    anchor_num = len(anchor_points)
    self.state_num = math.floor(1 / self.delt_val)
    self.anchor_points = anchor_points
    self.anchor_num = anchor_num
    self.max_time_stamp = 0.5 * self.anchor_num ** 2 if max_time_stamp <= 0 else max_time_stamp
    self.action_fifo_len = action_fifo_len
    self.basePath = basePath

    self.reset()
    super().__init__()

  def states(self):
    """
    According to code in `~/miniconda3/envs/py36/lib/python3.6/site-packages/tensorforce/core/utils/tensor_spec.py`
    num_values and min/max_value can't exist both
    if type is 'int', you have to specify num_values
    if you want to specify min/max value, the type has to be 'float'
    """
    return dict(
      line=dict(type='int', shape=(2,), num_values=self.anchor_num),
      img=dict(type='float', shape=self.img_shape + (3,),
               min_value=-1.0, max_value=1.0))
    # return dict(type='float', shape=self.img_shape + (2,),
    #             min_value=-1.0, max_value=1.0)

  def actions(self):
    """Action from 0 to anchor number-1, means drawing line from loc1 to next loc
    then set loc0 = loc1, loc1 = next_position
    if `widthdraw` > 0, then clear the earliest line
    """
    return dict(
      anchor_from=dict(type='int', num_values=self.anchor_num),
      anchor_to=dict(type='int', num_values=self.anchor_num))

  # Optional, should only be defined if environment has a natural maximum
  # episode length
  def max_episode_timesteps(self):
    return None

  # Optional
  def close(self):
    super().close()

  def reset(self):
    """Reset state.
    """
    self.timestep = 0
    self.last_action = 0
    self.action_fifo = deque(maxlen=self.state_num * 2)
    self.reward_fifo = deque(maxlen=500)
    self.canvas = Canvas(self.img_shape[1], self.img_shape[0], np.float)
    self.states_counter = np.zeros(shape=(self.anchor_num, self.anchor_num), dtype=np.float)
    return dict(img=np.zeros(shape=self.img_shape + (3,), dtype=np.float),
                line=[0, 0])

  def response(self, anchor_from, anchor_to):
    self.action_fifo.append([anchor_from, anchor_to])
    last_canvas = self.canvas.get_img().copy()
    self.canvas.line(self.anchor_points[anchor_from],
                     self.anchor_points[anchor_to],
                     self.delt_val)
    self.states_counter[anchor_from, anchor_to] += 1
    self.states_counter[anchor_to, anchor_from] += 1
    return np.clip(self.canvas.get_img(), 0, 1), np.clip(last_canvas, 0, 1)

  def reward_compute(self, line, canvas):
    line_mask = (line > 0) + 0.0
    diff_abs = np.abs((self.ref_image - canvas) * line_mask)
    reward_pos = self.gauss_weight * np.square(line_mask * (1 - diff_abs)) * self.mask_img
    reward_neg = self.gauss_weight * np.square(diff_abs) * self.mask_img_neg * self.neg_discount
    reward = np.sum(reward_pos) - np.sum(reward_neg)
    self.reward_fifo.append(reward)
    return reward

  def check_no_progress_recent(self):
    if self.timestep < self.max_time_stamp / 3:
      return False
    r0 = self.reward_fifo[0]
    for r in self.reward_fifo:
      if r > r0:
        return False
      r0 = r
    return True

  def form_action_mask(self, anchor_to):
    from_mask = [False] * self.anchor_num
    to_mask = [True] * self.anchor_num
    # force next from anchor to be `anchor_to` of last action
    from_mask[anchor_to] = True
    to_mask[anchor_to] = False
    # rule2: prevent draw lines >= self.state_num between two points
    bold_line = np.argwhere(self.states_counter >= self.state_num / 10)
    for p in bold_line:
      if anchor_to == p[0]:
        to_mask[p[1]] = False
      elif anchor_to == p[1]:
        to_mask[p[0]] = False
    return from_mask, to_mask

  def execute(self, actions):
    if self.timestep % 100 == 0:
      print(f"{datetime.now()}: step {self.timestep}: action = {actions}")

    anchor_from, anchor_to = actions['anchor_from'], actions['anchor_to']
    if self.timestep % 100 == 0:
      self.canvas.show(path.join(self.basePath, f"temp_{self.timestep-1}.jpg"))
    ## Update the current canvas
    canvas, last_canvas = self.response(anchor_from, anchor_to)
    new_line = canvas - last_canvas
    ## Compute the reward
    reward = self.reward_compute(new_line, canvas)

    if self.timestep % 100 == 0:
      print(f"{datetime.now()}: step {self.timestep}: reward = {reward}, recent actions: {self.action_fifo}")
      self.canvas.show(path.join(self.basePath, f"temp_{self.timestep}.jpg"))
      # if reward > -0.03:
      #     self.canvas.show(path.join(self.basePath, f"paint_result_{self.timestep}.jpg"))
    ## Increment timestamp
    self.timestep += 1

    ## The only way to go terminal is to exceed max_episode_timestamp.
    ## terminal == False means episode is not done
    ## terminal == True means it is done.
    terminal = False
    if self.timestep > self.max_time_stamp:
      terminal = True
      print(f"{datetime.now()}: terminal at max timestamp {self.timestep}, action = {actions}, reward = {reward}")

    # if self.check_no_progress_recent():
    #   terminal = True
    #   print(
    #     f"{datetime.now()}: terminal by no progress recently, at {self.timestep}, action = {actions}, reward = {reward}")

    from_mask, to_mask = self.form_action_mask(anchor_to)

    # find next valid line
    if not any(to_mask):
      valid_line_num = np.sum(self.states_counter, 0)
      to2 = np.argmin(valid_line_num)
      from_mask, to_mask = self.form_action_mask(to2)
      if not any(to_mask):
        print(f"{datetime.now()}: invalid start point:{anchor_to}, terminate because no valid actions, {valid_line_num}")
        from_mask = to_mask = [True] * self.anchor_num
        terminal = True
      else:
        print(f"{datetime.now()}: invalid start point:{anchor_to}, find a new anchor to start a line {to2}, {valid_line_num}")

    return dict(line=[anchor_from, anchor_to],
                img=np.stack([new_line, last_canvas, self.ref_image], axis=-1),
                anchor_from_mask=from_mask,
                anchor_to_mask=to_mask), terminal, reward


###-----------------------------------------------------------------------------
### Create the environment
###   - Tell it the environment class
###   - Set the max timestamps that can happen per episode
if __name__ == "__main__":
  ref_img = Canvas(11, 11, np.float)
  ref_img.circle(Point(5, 5), 1, 3)
  envn = DrawingEnvironment(1 - ref_img.get_img(), 0.3, [])
  canvas1 = Canvas(11, 11, np.float)
  canvas1.line(Point(4, 0), Point(4, 10), 0.3)
  canvas1.line(Point(7, 0), Point(7, 10), 0.3)
  r1 = envn.reward_compute(canvas1.get_img())
  canvas1 = Canvas(11, 11, np.float)
  canvas1.line(Point(5, 0), Point(5, 10), 0.3)
  canvas1.line(Point(0, 5), Point(10, 5), 0.3)
  r2 = envn.reward_compute(canvas1.get_img())
  print(f"r1 = {r1}, r2 = {r2}")
