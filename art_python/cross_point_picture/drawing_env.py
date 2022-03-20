###-----------------------------------------------------------------------------
## 
from canvas import Canvas, Point
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
    def __init__(self, ref_image, delt_val:float, anchor_points:list):
        self.ref_image = ref_image
        self.img_shape = ref_image.shape
        self.delt_val = delt_val
        anchor_num = len(anchor_points)
        self.state_num = math.floor(1/self.delt_val)
        self.anchor_points = anchor_points
        self.anchor_num = anchor_num
        self.reset()
        super().__init__()


    def states(self):
        """
        According to code in `~/miniconda3/envs/py36/lib/python3.6/site-packages/tensorforce/core/utils/tensor_spec.py`
        num_values and min/max_value can't exist both  
        """
        return dict(type='int', shape=(self.anchor_num, self.anchor_num), num_values=self.state_num+1)
        # , min_value=0, max_value=self.state_num)

    def actions(self):
        """Action from 0 to anchor number-1, means drawing line from loc1 to next loc
        then set loc0 = loc1, loc1 = next_position
        """
        return dict(type='float64', shape=(self.num_actors,), min_value=0.0, max_value=1.0)

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return self.state_num*self.anchor_num

    # Optional
    def close(self):
        super().close()

    def reset(self):
        """Reset state.
        """
        self.timestep = 0
        self.loc1 = None
        self.last_action = -1
        self.canvas = Canvas(self.img_shape[0], self.img_shape[1], np.float64)
        self.states_mat = np.zeros(shape=(self.anchor_num, self.anchor_num), dtype=np.int)
        return self.states_mat


    def response(self, action_prob):
        action = np.argmax(action_prob)
        if self.loc1 is not None:
            self.canvas.line(self.loc1, self.anchor_points[action], self.delt_val)
            self.states_mat[self.last_action, action] += 1 
            # self.states_mat = np.clip(self.states_mat, 0, self.state_num)
        self.loc1 = self.anchor_points[action]
        self.last_action = action

    def reward_compute(self):
        return -np.sum(np.abs(self.ref_image - 
                                 self.canvas.get_img()))

    def execute(self, actions):
        ## Increment timestamp
        self.timestep += 1
        ## Update the current canvas
        self.response(actions)
        ## Compute the reward
        reward = self.reward_compute()

        if self.timestep % 100 == 0:
            print(f"{self.timestep}: action = {actions}, reward = {reward}")

        ## The only way to go terminal is to exceed max_episode_timestamp.
        ## terminal == False means episode is not done
        ## terminal == True means it is done.
        terminal = False
        if self.timestep > self.state_num*self.anchor_num:
            terminal = True
            print(f"terminal at timestamp: {self.timestep}, action = {actions}, reward = {reward}")
        return self.states_mat, terminal, reward

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
