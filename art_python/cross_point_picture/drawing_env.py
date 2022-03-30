###-----------------------------------------------------------------------------
## 
from collections import deque
from canvas import Canvas, Point, Mode
from datetime import datetime
import math
import numpy as np
from tensorforce.environments import Environment
from tensorforce.agents import Agent


###-----------------------------------------------------------------------------
### Environment definition
class DrawingEnvironment(Environment):
    """This class defines a simple drawing environment. 
    The award is the image diff value between current image the reference image
    ref_img: float matrix, gray value in [0, 1]
    """

    def __init__(self, ref_image, delt_val: float, anchor_points: list, max_time_stamp: int = -1,
                 action_fifo_len: int = 15):
        self.ref_image = ref_image.astype(np.float)
        self.img_shape = ref_image.shape
        self.delt_val = delt_val
        anchor_num = len(anchor_points)
        self.state_num = math.floor(1 / self.delt_val)
        self.anchor_points = anchor_points
        self.anchor_num = anchor_num
        self.max_time_stamp = self.state_num * self.anchor_num if max_time_stamp <= 0 else max_time_stamp
        self.action_fifo_len = action_fifo_len
        self.reset()
        super().__init__()

    def states(self):
        """
        According to code in `~/miniconda3/envs/py36/lib/python3.6/site-packages/tensorforce/core/utils/tensor_spec.py`
        num_values and min/max_value can't exist both  
        if type is 'int', you have to specify num_values, so can only make states as float
        """
        return dict(type='float', shape=self.img_shape + (1,),
                    min_value=-1.0, max_value=1.0)

    def actions(self):
        """Action from 0 to anchor number-1, means drawing line from loc1 to next loc
        then set loc0 = loc1, loc1 = next_position
        if `widthdraw` > 0, then clear the earliest line
        """
        return dict(
            anchor=dict(type='int', num_values=self.anchor_num),
            withdw=dict(type='float', min_value=-1.0, max_value=1.0)
        )

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
        self.action_history = deque()
        self.canvas = Canvas(self.img_shape[0], self.img_shape[1], np.float)
        self.states_counter = np.zeros(shape=self.img_shape+(1,), dtype=np.float)
        return self.states_counter

    def response(self, actions):
        anchor_ind, widthdraw = actions[0], actions[1]
        if widthdraw > 0 and len(self.action_history) > self.action_fifo_len:
            rm_from = self.action_history.popleft()
            rm_to = self.action_history[0]
            assert self.states_counter[rm_from, rm_to] > 0
            rm_from_point = self.anchor_points[rm_from]
            rm_to_point = self.anchor_points[rm_to]
            self.states_counter[rm_from, rm_to] -= 1
            self.canvas.line(rm_from_point, rm_to_point, self.delt_val, Mode.Subtract)
        else:
            self.action_fifo.append(anchor_ind)
            self.action_history.append(anchor_ind)
            if self.loc1 is not None:
                self.canvas.line(self.loc1, self.anchor_points[anchor_ind], self.delt_val)
                self.states_counter[self.last_action, anchor_ind] += 1
                self.states_counter = np.clip(self.states_counter, 0., self.state_num)
            self.loc1 = self.anchor_points[anchor_ind]
            self.last_action = anchor_ind

    # only care about the minimum top 10% diff
    def reward_compute(self):
        diff = self.ref_image - self.canvas.get_img()
        diff_abs = np.abs(diff)
        cnt, bar = np.histogram(diff_abs, bins=10)
        mask_bool = diff_abs >= bar[1]
        diff[mask_bool] = 0
        diff_abs[mask_bool] = 0
        return diff, -np.sum(diff_abs)/cnt[0]

    def execute(self, actions):
        ## Update the current canvas
        self.response(actions)
        ## Compute the reward
        diff_img, reward = self.reward_compute()

        if self.timestep % 10 == 0:
            print(f"{datetime.now()}: {self.timestep}: action = {actions}, reward = {reward}")
        ## Increment timestamp
        self.timestep += 1

        anchor_ind = actions[0]
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

        return dict(state=np.expand_dims(diff_img, axis=-1),
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
