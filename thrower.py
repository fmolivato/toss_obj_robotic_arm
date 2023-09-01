import numpy as np

from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco as mj


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "lookat": [0, 0, 0],
    "distance": 6.0,
    "azimuth": 270,
    "elevation": -30,
}


class ThrowerEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description
    "Pusher" is a multi-jointed robot arm which is very similar to that of a human.
     The goal is to move a target cylinder (called *object*) to a goal position using the robot's end effector (called *fingertip*).
      The robot consists of shoulder, elbow, forearm, and wrist joints.

    ## Action Space
    The action space is a `Box(-2, 2, (7,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.

    | Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    |-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0    | Rotation of the panning the shoulder                              | -2          | 2           | r_shoulder_pan_joint             | hinge | torque (N m) |
    | 1    | Rotation of the shoulder lifting joint                            | -2          | 2           | r_shoulder_lift_joint            | hinge | torque (N m) |
    | 2    | Rotation of the shoulder rolling joint                            | -2          | 2           | r_upper_arm_roll_joint           | hinge | torque (N m) |
    | 3    | Rotation of hinge joint that flexed the elbow                     | -2          | 2           | r_elbow_flex_joint               | hinge | torque (N m) |
    | 4    | Rotation of hinge that rolls the forearm                          | -2          | 2           | r_forearm_roll_joint             | hinge | torque (N m) |
    | 5    | Rotation of flexing the wrist                                     | -2          | 2           | r_wrist_flex_joint               | hinge | torque (N m) |
    | 6    | Rotation of rolling the wrist                                     | -2          | 2           | r_wrist_roll_joint               | hinge | torque (N m) |

    ## Observation Space

    Observations consist of

    - Angle of rotational joints on the pusher
    - Angular velocities of rotational joints on the pusher
    - The coordinates of the fingertip of the pusher
    - The coordinates of the object to be moved
    - The coordinates of the goal position

    The observation is a `ndarray` with shape `(23,)` where the elements correspond to the table below.
    An analogy can be drawn to a human arm in order to help understand the state space, with the words flex and roll meaning the
    same as human joints.

    | Num | Observation                                              | Min  | Max | Name (in corresponding XML file) | Joint    | Unit                     |
    | --- | -------------------------------------------------------- | ---- | --- | -------------------------------- | -------- | ------------------------ |
    | 0   | Rotation of the panning the shoulder                     | -Inf | Inf | r_shoulder_pan_joint             | hinge    | angle (rad)              |
    | 1   | Rotation of the shoulder lifting joint                   | -Inf | Inf | r_shoulder_lift_joint            | hinge    | angle (rad)              |
    | 2   | Rotation of the shoulder rolling joint                   | -Inf | Inf | r_upper_arm_roll_joint           | hinge    | angle (rad)              |
    | 3   | Rotation of hinge joint that flexed the elbow            | -Inf | Inf | r_elbow_flex_joint               | hinge    | angle (rad)              |
    | 4   | Rotation of hinge that rolls the forearm                 | -Inf | Inf | r_forearm_roll_joint             | hinge    | angle (rad)              |
    | 5   | Rotation of flexing the wrist                            | -Inf | Inf | r_wrist_flex_joint               | hinge    | angle (rad)              |
    | 6   | Rotation of rolling the wrist                            | -Inf | Inf | r_wrist_roll_joint               | hinge    | angle (rad)              |
    | 7   | Rotational velocity of the panning the shoulder          | -Inf | Inf | r_shoulder_pan_joint             | hinge    | angular velocity (rad/s) |
    | 8   | Rotational velocity of the shoulder lifting joint        | -Inf | Inf | r_shoulder_lift_joint            | hinge    | angular velocity (rad/s) |
    | 9   | Rotational velocity of the shoulder rolling joint        | -Inf | Inf | r_upper_arm_roll_joint           | hinge    | angular velocity (rad/s) |
    | 10  | Rotational velocity of hinge joint that flexed the elbow | -Inf | Inf | r_elbow_flex_joint               | hinge    | angular velocity (rad/s) |
    | 11  | Rotational velocity of hinge that rolls the forearm      | -Inf | Inf | r_forearm_roll_joint             | hinge    | angular velocity (rad/s) |
    | 12  | Rotational velocity of flexing the wrist                 | -Inf | Inf | r_wrist_flex_joint               | hinge    | angular velocity (rad/s) |
    | 13  | Rotational velocity of rolling the wrist                 | -Inf | Inf | r_wrist_roll_joint               | hinge    | angular velocity (rad/s) |
    | 14  | x-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                         | slide    | position (m)             |
    | 15  | y-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                         | slide    | position (m)             |
    | 16  | z-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                         | slide    | position (m)             |
    | 17  | x-coordinate of the object to be moved                   | -Inf | Inf | object (obj_slidex)              | slide    | position (m)             |
    | 18  | y-coordinate of the object to be moved                   | -Inf | Inf | object (obj_slidey)              | slide    | position (m)             |
    | 19  | z-coordinate of the object to be moved                   | -Inf | Inf | object                           | cylinder | position (m)             |
    | 20  | x-coordinate of the goal position of the object          | -Inf | Inf | goal (goal_slidex)               | slide    | position (m)             |
    | 21  | y-coordinate of the goal position of the object          | -Inf | Inf | goal (goal_slidey)               | slide    | position (m)             |
    | 22  | z-coordinate of the goal position of the object          | -Inf | Inf | goal                             | sphere   | position (m)             |


    ## Rewards
    The reward consists of two parts:
    - *reward_near *: This reward is a measure of how far the *fingertip*
    of the pusher (the unattached end) is from the object, with a more negative
    value assigned for when the pusher's *fingertip* is further away from the
    target. It is calculated as the negative vector norm of (position of
    the fingertip - position of target), or *-norm("fingertip" - "target")*.
    - *reward_dist *: This reward is a measure of how far the object is from
    the target goal position, with a more negative value assigned for object is
    further away from the target. It is calculated as the negative vector norm of
    (position of the object - position of goal), or *-norm("object" - "target")*.
    - *reward_control*: A negative reward for penalising the pusher if
    it takes actions that are too large. It is measured as the negative squared
    Euclidean norm of the action, i.e. as *- sum(action<sup>2</sup>)*.

    The total reward returned is ***reward*** *=* *reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near*

    Unlike other environments, Pusher does not allow you to specify weights for the individual reward terms.
    However, `info` does contain the keys *reward_dist* and *reward_ctrl*. Thus, if you'd like to weight the terms,
    you should create a wrapper that computes the weighted reward from `info`.


    ## Starting State
    All pusher (not including object and goal) states start in
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0). A uniform noise in the range
    [-0.005, 0.005] is added to the velocity attributes only. The velocities of
    the object and goal are permanently set to 0. The object's x-position is selected uniformly
    between [-0.3, 0] while the y-position is selected uniformly between [-0.2, 0.2], and this
    process is repeated until the vector norm between the object's (x,y) position and origin is not greater
    than 0.17. The goal always have the same position of (0.45, -0.05, -0.323).

    The default framerate is 5 with each frame lasting for 0.01, giving rise to a *dt = 5 * 0.01 = 0.05*

    ## Episode End

    The episode ends when any of the following happens:

    1. Truncation: The episode duration reaches a 100 timesteps.
    2. Termination: Any of the state space values is no longer finite.

    ## Arguments

    No additional arguments are currently supported (in v2 and lower),
    but modifications can be made to the XML file in the assets folder
    (or by changing the path to a modified XML file in another folder)..

    ```python
    import gymnasium as gym
    env = gym.make('Pusher-v4')
    ```

    There is no v3 for Pusher, unlike the robot environments where a v3 and
    beyond take `gymnasmium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc.

    ```python
    import gymnasium as gym
    env = gym.make('Pusher-v2')
    ```

    ## Version History

    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3
    * v2: All continuous control environments now use mujoco-py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks (not including reacher, which has a max_time_steps of 50). Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        # observation_space = Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64)

        observation_space = spaces.Dict(
            dict(
                # desired_goal=spaces.Box(
                #     -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                # ),
                # achieved_goal=spaces.Box(
                #     -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                # ),
                observation=spaces.Box(-np.inf, np.inf, shape=(23,), dtype="float64"),
            )
        )

        MujocoEnv.__init__(
            self,
            "/home/serafino-pc/Documenti/Francesco/unimore/smart_robotics/lab/thrower.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.current_step = 0
        self.cur_state = None  # curriculum state

        # curriculum learning triggers
        self.cur = {
            "goal_static": 0,
            "goal_rnd_near": 192 * 62500,
            "goal_rnd_far": 192 * 62500 * 2,
        }

        self.cur_steps_triggers = np.array(list(self.cur.values()))
        self.cur_steps_triggers_range = np.arange(self.cur_steps_triggers.shape[0])
        self.cur_states = list(self.cur.keys())

    def step(self, a):
        # counting steps
        self.current_step += 1

        # reward
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_dist = -np.linalg.norm(vec_2)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + 0.1 * reward_ctrl

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        return (
            ob,
            reward,
            False,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

    def reset(self):
        # check curriculum state
        self._update_curriculum_state()

        return self.reset_model(), {}

    def reset_model(self):
        qpos = self.init_qpos

        # set initial goal position
        # if "goal_static" == self.cur_state or self.cur_state is None:
        #     self.goal_pos = np.zeros((3,))

        # elif "goal_rnd_near" == self.cur_state:
        #     self.goal_pos = np.concatenate(
        #         [
        #             self.np_random.uniform(low=-0.5, high=0.5, size=1),
        #             self.np_random.uniform(low=0, high=0.5, size=1),
        #             np.zeros((1,)),
        #         ]
        #     )

        # elif "goal_rnd_far" == self.cur_state:
        #     self.goal_pos = np.concatenate(
        #         [
        #             self.np_random.uniform(low=-1, high=1, size=1),
        #             self.np_random.uniform(low=0, high=1, size=1),
        #             np.zeros((1,)),
        #         ]
        #     )
        self.goal_pos = np.concatenate(
            [
                self.np_random.uniform(low=-1, high=1, size=1),
                self.np_random.uniform(low=0, high=1, size=1),
                np.zeros((1,)),
            ]
        )

        # if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.04:
        #     break

        # set initial cylinder position
        self.cylinder_pos = np.asarray([0, 0, 0])

        qpos[-5:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos[:-1]
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        # curriculum using for masking the observations
        joint_pos = self.data.qpos.flat[:7]  # * self._get_obs_mask()
        joint_vel = self.data.qvel.flat[:7]  # * self._get_obs_mask()

        obs = np.concatenate(
            [
                joint_pos,
                joint_vel,
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )

        return {"observation": obs}

    def _get_obs_mask(self):
        """Used for curriculum learning"""
        mask = np.asarray([1, 1, 1, 1, 1, 1, 1])

        return mask

    def _update_curriculum_state(self):
        current_total_steps = self.current_step * 16  # env batch
        future_steps = self.cur_steps_triggers_range[
            self.cur_steps_triggers <= current_total_steps
        ]
        cur_state_idx = future_steps[-1]

        # if self.cur_state != self.cur_states[cur_state_idx]:
        self.cur_state = self.cur_states[cur_state_idx]

        print(f"{current_total_steps=}")
        print(f"{self.cur_steps_triggers=}")
        print(f"{future_steps=}")
        print(f"{cur_state_idx=}")
        print(f"{self.cur_state=}")
