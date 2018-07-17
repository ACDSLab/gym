import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class ArmEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    FRAME_SKIP = 2
    JOINTS = ['wam/' + joint for joint in ['base_yaw_joint',
                                           'shoulder_pitch_joint',
                                           'shoulder_yaw_joint',
                                           'elbow_pitch_joint',
                                           'wrist_yaw_joint',
                                           'wrist_pitch_joint',
                                           'palm_yaw_joint']]

    JOINT_OFFSET = {JOINTS[1]: 1.0,
                    JOINTS[3]: 2.1,
                    JOINTS[5]: 0.0}

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm/arm.xml', self.FRAME_SKIP)
        utils.EzPickle.__init__(self)

    def step(self, a):
        # posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        # posafter, height, ang = self.sim.data.qpos[0:3]
        # alive_bonus = 1.0
        # reward = (posafter - posbefore) / self.dt
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()
        # s = self.state_vector()
        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
        #             (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        reward = 0
        return ob, reward, False, {}

    def _get_obs(self):
        qpos = np.array([self.sim.data.get_joint_qpos(joint) for joint in self.JOINTS])
        qvel = np.array([self.sim.data.get_joint_qvel(joint) for joint in self.JOINTS])
        return np.concatenate([qpos, qvel])

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        # qpos = self.init_qpos + self.np_random.uniform(low=-.000, high=.005, size=self.model.nq)
        # qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        for joint, offset in self.JOINT_OFFSET.items():
            pos = self.sim.data.get_joint_qpos(joint)
            self.sim.data.set_joint_qpos(joint, pos + offset)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = 1.75
        self.viewer.cam.elevation = -15
        self.viewer.cam.azimuth = 135
