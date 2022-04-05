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
    x, y = np.mgrid[-size[0]//2 + 1:size[0]//2 + 1, -size[1]//2 + 1:size[1]//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
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
                basePath = "."):
        self.ref_image = ref_image.astype(np.float)
        self.img_shape = ref_image.shape
        self.delt_val = delt_val
        self.contrast = contrast

        self.gauss_mask = gauss_mask(self.img_shape, np.max(self.img_shape)*0.8)
        self.mask_img = self.ref_image > self.contrast
        self.mask_img_neg = np.logical_not(self.mask_img)
        self.mask_sum = np.sum(self.mask_img)
        self.mask_sum_neg = ref_image.size - self.mask_sum
        self.neg_discount = self.mask_sum_neg / self.mask_sum
        print(f"mask sum pos = {self.mask_sum}, neg = {self.mask_sum_neg}")
        anchor_num = len(anchor_points)
        self.state_num = math.floor(1 / self.delt_val)
        self.anchor_points = anchor_points
        self.anchor_num = anchor_num
        self.max_time_stamp = self.state_num * self.anchor_num if max_time_stamp <= 0 else max_time_stamp
        self.action_fifo_len = action_fifo_len
        self.basePath = basePath

        self.reset()
        super().__init__()

    def states(self):
        """
        According to code in `~/miniconda3/envs/py36/lib/python3.6/site-packages/tensorforce/core/utils/tensor_spec.py`
        num_values and min/max_value can't exist both
        if type is 'int', you have to specify num_values, so can only make states as float
        """
        return dict(type='float', shape=self.img_shape + (2,),
                    min_value=-1.0, max_value=1.0)

    def actions(self):
        """Action from 0 to anchor number-1, means drawing line from loc1 to next loc
        then set loc0 = loc1, loc1 = next_position
        if `widthdraw` > 0, then clear the earliest line
        """
        return dict(
            anchor=dict(type='int', num_values=self.anchor_num))

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
        self.loc1 = None
        self.last_action = -1
        self.action_fifo = deque(maxlen=self.action_fifo_len)
        self.canvas = Canvas(self.img_shape[1], self.img_shape[0], np.float)
        self.states_counter = np.zeros(shape=(self.anchor_num, self.anchor_num), dtype=np.float)
        return np.zeros(shape=self.img_shape + (2,), dtype=np.float)

    def response(self, actions):
        anchor_ind = actions['anchor']
        self.action_fifo.append(anchor_ind)
        if self.loc1 is not None:
            self.canvas.line(self.loc1, self.anchor_points[anchor_ind], self.delt_val)
            self.states_counter[self.last_action, anchor_ind] += 1
        self.loc1 = self.anchor_points[anchor_ind]
        self.last_action = anchor_ind

    # only care about the minimum top 10% diff
    def reward_compute(self):
        canvas = self.canvas.get_img()
        minv, maxv = np.min(canvas), np.max(canvas)
        canvas = (canvas - minv)/(maxv - minv + 1e-5)
        diff = self.gauss_mask * (self.ref_image - canvas)
        diff = np.clip(diff, a_min=-1., a_max=1.)
        pos_diff = np.sum(diff[self.mask_img]) / self.mask_sum
        neg_diff = np.sum(diff[self.mask_img_neg]) / self.mask_sum_neg
        return diff, - pos_diff - neg_diff*self.neg_discount

    def execute(self, actions):
        if self.timestep % 100 == 0:
            print(f"{datetime.now()}: step {self.timestep}: action = {actions}")
        ## Update the current canvas
        self.response(actions)
        ## Compute the reward
        diff_img, reward = self.reward_compute()

        if self.timestep % 100 == 0:
            print(f"{datetime.now()}: step {self.timestep}: reward = {reward}, recent actions: {self.action_fifo}")
            self.canvas.show(path.join(self.basePath, "temp.jpg"))
        ## Increment timestamp
        self.timestep += 1

        anchor_ind = actions['anchor']
        ## The only way to go terminal is to exceed max_episode_timestamp.
        ## terminal == False means episode is not done
        ## terminal == True means it is done.
        terminal = False
        if self.timestep > self.max_time_stamp:
            terminal = True
            print(f"{datetime.now()}: terminal at timestamp {self.timestep}, action = {actions}, reward = {reward}")

        action_mask = [True] * self.anchor_num
        # rule1: prevent sink to one point
        action_mask[anchor_ind] = False
        # rule2: prevent draw lines >= self.state_num between two points
        bold_line = np.argwhere((self.states_counter + np.transpose(self.states_counter))
                                >= self.state_num)
        for p in bold_line:
            if anchor_ind == p[0]:
                action_mask[p[1]] = False
            elif anchor_ind == p[1]:
                action_mask[p[0]] = False
        # rule3: prevent recent action
        for a in self.action_fifo:
            action_mask[a] = False

        return dict(state=np.stack([diff_img, self.ref_image], axis=-1),
                    anchor_mask=action_mask), terminal, reward


###-----------------------------------------------------------------------------
### Create the environment
###   - Tell it the environment class
###   - Set the max timestamps that can happen per episode
if __name__ == "__main__":
    ref_img = np.zeros((100, 100), np.uint8)
    envn = DrawingEnvironment(ref_img, 50, [
        Point(2, 3), Point(99, 99),
        Point(8, 3), Point(1, 9)
    ])
    print(f"actions: {envn.actions()}")
    print(f"states: {envn.states()}")
    print("step:", envn.execute(1))
    print("step:", envn.execute(2))
