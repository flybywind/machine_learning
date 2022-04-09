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

        self.gauss_mask = gauss_mask(self.img_shape, np.max(self.img_shape)*0.5)
        self.mask_img = self.ref_image > self.contrast
        self.mask_img_neg = np.logical_not(self.mask_img)
        self.mask_sum = np.sum(self.mask_img)
        self.mask_sum_neg = ref_image.size - self.mask_sum
        self.neg_discount = self.mask_sum / self.mask_sum_neg
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
        if type is 'int', you have to specify num_values
        if you want to specify min/max value, the type has to be 'float'
        """
        return dict(
            line=dict(type='int', shape=(2,), num_values=self.anchor_num),
            img=dict(type='float', shape=self.img_shape + (2,),
                    min_value=-1.0, max_value=1.0))
        # return dict(type='float', shape=self.img_shape + (2,),
        #             min_value=-1.0, max_value=1.0)

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
        self.last_action = 0
        self.action_fifo = deque(maxlen=self.state_num*2)
        self.reward_fifo = deque(maxlen=500)
        self.canvas = Canvas(self.img_shape[1], self.img_shape[0], np.float)
        self.states_counter = np.zeros(shape=(self.anchor_num, self.anchor_num), dtype=np.float)
        return dict(img=np.zeros(shape=self.img_shape + (2,), dtype=np.float),
                    line=[0, 0])

    def response(self, actions):
        anchor_ind = actions['anchor']
        self.action_fifo.append(anchor_ind)
        last_canvas = self.canvas.get_img().copy()
        if self.loc1 is not None:
            self.canvas.line(self.loc1, self.anchor_points[anchor_ind], self.delt_val)
            self.states_counter[self.last_action, anchor_ind] += 1
        self.loc1 = self.anchor_points[anchor_ind]
        self.last_action = anchor_ind
        return self.canvas.get_img(), last_canvas

    def reward_compute(self, last_canvas, canvas):
        last_can_clip = np.clip(last_canvas, 0., 1.)
        canvas_clip = np.clip(canvas, 0., 1.)
        diff = self.gauss_mask * (self.ref_image - canvas_clip)
        diff0 = self.gauss_mask * (self.ref_image - last_can_clip)
        new_line = canvas_clip - last_can_clip
        line_mask = new_line > 0
        # point reward: ie, if the line points fall into the target image area, we give them a
        # positive reward, else for points out of target image, negative reward
        reward = np.sum((np.abs(diff0) - np.abs(diff)) * line_mask)

        # cross point reward: ie, when the newly painted line cross with other lines, if the cross points
        # fall into the target image, we give them a positive reward, else for crosses out of target image,
        # we give them negative reward
        new_line = canvas - last_canvas
        cross_point = np.logical_and(last_canvas > 0., new_line > 0.)
        cp_pos = np.logical_and(cross_point, self.mask_img)
        cp_neg = np.logical_and(cross_point, self.mask_img_neg)
        reward2 = (np.sum(cp_pos) - np.sum(cp_neg))*self.neg_discount
        reward += reward2
        self.reward_fifo.append(reward)
        diff = np.clip(diff, a_min=-1., a_max=1.)
        return diff, reward

    def check_no_progress_recent(self):
        if self.timestep < self.max_time_stamp/3:
            return False
        r0 = self.reward_fifo[0]
        for r in self.reward_fifo:
            if r > r0:
                return False
            r0 = r
        return True

    def execute(self, actions):
        if self.timestep % 100 == 0:
            print(f"{datetime.now()}: step {self.timestep}: action = {actions}")

        pre_anchor = self.last_action
        anchor_ind = actions['anchor']
        ## Update the current canvas
        canvas, last_canvas = self.response(actions)
        new_line = canvas - last_canvas
        ## Compute the reward
        diff_img, reward = self.reward_compute(last_canvas, canvas)

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

        if self.check_no_progress_recent():
            terminal = True
            print(f"{datetime.now()}: terminal by no progress recently, at {self.timestep}, action = {actions}, reward = {reward}")

        action_mask = [True] * self.anchor_num
        # rule1: prevent sink to one point
        action_mask[anchor_ind] = False
        # rule2: prevent draw lines >= self.state_num between two points
        bold_line = np.argwhere((self.states_counter + np.transpose(self.states_counter))
                                == self.state_num)
        invalid_line = np.argwhere((self.states_counter + np.transpose(self.states_counter))
                                > self.state_num)
        assert len(invalid_line) == 0

        for p in bold_line:
            if anchor_ind == p[0]:
                action_mask[p[1]] = False
            elif anchor_ind == p[1]:
                action_mask[p[0]] = False
        # # rule3: prevent recent action
        # for a in self.action_fifo:
        #     action_mask[a] = False

        return dict(line=[pre_anchor, anchor_ind],
            img=np.stack([diff_img, new_line], axis=-1),
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
