from dataclasses import dataclass, field
import numpy as np
# import torch
from scipy.spatial.transform import Rotation

from env.mocks import MockGripper, MockRobot
from env.lift import LiftEEConfig
from env.drawer import DrawerEEConfig
from env.hang import HangEEConfig
from env.towel import TowelEEConfig
from env.two_stage import TwoStageEEConfig

# from robots.ur import URRobot
from zmq_core.robot_node import ZMQClientRobot
# from env.env import RobotEnv
from robots.robotiq_gripper import RobotiqGripper
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import pyrallis
try:
    import rtde_control
    import rtde_receive

    URTDE_IMPORTED = True
except ImportError:
    print("[robots] Skipping URTDE")
    URTDE_IMPORTED = False


class ActionSpace:
    def __init__(self, low: list[float], high: list[float]):
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)

    def assert_in_range(self, actionl: list[float]):
        action: np.ndarray = np.array(actionl)

        correct = (action <= self.high).all() and (action >= self.low).all()
        if correct:
            return True

        for i in range(self.low.size):
            check = action[i] >= self.low[i] and action[i] <= self.high[i]
            print(f"{self.low[i]:.2f} <= {action[i]:.2f} <= {self.high[i]:.2f}? {check}")
        return False

    def clip_action(self, action: np.ndarray):
        clipped = np.clip(action, self.low, self.high)
        return clipped


@dataclass
class URTDEControllerConfig:
    task: str = "lift"

    # robot_ip_address: str = "localhost"
    controller_type: str = "CARTESIAN_DELTA"
    max_delta: float = 0.06
    mock: int = 0

    ## --------- Newly Added ----------- ##
    agent: str = "ur"
    hostname: str = "10.104.192.228"
    # hostname: str = "192.168.77.243"
    robot_port: int = 50003  # for trajectory
    robot_ip: str = "192.168.77.21" 
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_pose: Optional[Tuple[float, ...]] = None

    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    verbose: bool = False
    camera_clients = {
    # you can optionally add camera nodes here for imitation learning purposes
    # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
    # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
    }




class URTDEController:
    """Controller run on server.

    All parameters should be python native for easy integration with 0rpc
    """

    def __init__(self, cfg: URTDEControllerConfig, task) -> None:
        print("URTDE Controller Initialized...")
    
        self.cfg = cfg
        assert self.cfg.controller_type in {
            "CARTESIAN_DELTA",
            "CARTESIAN_IMPEDANCE",
        }

        self.use_gripper = False


        if task == "lift":
            self.ee_config = LiftEEConfig()
        elif task == "drawer":  
            self.ee_config = DrawerEEConfig()
        elif task == "two_stage":
            self.ee_config = TwoStageEEConfig()
        elif task == "hang":
            self.ee_config = HangEEConfig()
        elif task == "towel":
            self.ee_config = TowelEEConfig()
        else:
            assert False, "unknown env"

        self.action_space = ActionSpace(*self.get_action_space())

        if self.cfg.mock:
            self._robot = MockRobot()
            self._gripper = MockGripper()
        else:
            assert URTDE_IMPORTED, "Attempted to load robot without URTDE package."
            robot_client = ZMQClientRobot(port=cfg.robot_port, host=cfg.hostname)
            self._robot = robot_client
            if self.use_gripper:
                self._gripper = RobotiqGripper()
                self._gripper.connect(hostname=cfg.robot_ip, port=63352)
                print("gripper connected")


        print("Setting Home Position..")
        self.home_pose = self.ee_config.home
        print("Going to Home Position..")
        self.go_home(blocking=False)
        print(f"In home pose: joint_angles = {self.ee_config.home}")

        ee_pos = self._robot.get_ee_pose()
        print("current ee pos:", ee_pos)

        if self.use_gripper:
            if hasattr(self._gripper, "_max_position") :
                # Should grab this from robotiq2f
                self._max_gripper_width = self._gripper._max_position
            else:
                self._max_gripper_width = 255  # default, from Robotiq Value

        self.desired_gripper_qpos = 0

        self.reached_place = False
        self.reached_z_min = False

        ## --- Robot Workspace ---
        self.x_min, self.x_max = [-0.34, 0.32]
        self.y_min, self.y_max = [-0.453, -0.132]
        self.z_min, self.z_max = [-0.016, 0.267]    # Got by testing

        self.r_min, self.r_max = [2.477, 3.779]
        self.p_min, self.p_max = [-3.14, 3.14]
        self.zy_min, self.zy_max = [-1.219, 0.919]

        print("-------------------------------------")
        print("URTDE Controller Initialized...!!")
        print("-------------------------------------")

    def hello(self):
        return "hello"


   

    def go_home(self, blocking=True) -> None:
        """Calls move_to_joint_positions to the current home positions."""
        if blocking is False:
            assert (
                self.home_pose is not None
            ), "Home pose not assigned! Call 'set_home_pose(<joint_angles>)' to enable homing"

            if self.use_gripper:
                return self.move_to_joint_positions(positions=self.home_pose, delta=False)
            elif self.use_gripper is False:
                return self.move_to_joint_positions(positions=self.home_pose[:6], delta=False)
            else:
                print("Unknown Error occured")
                return         



    def move_to_joint_positions(self, positions: np.ndarray, delta: bool = False):
        """Moves the robot to the specified joint positions.

        Args:
            positions (torch.Tensor): The joint positions to move the robot to.
            delta (bool, optional): Whether the positions are relative to the current positions. Defaults to False.

        Returns:
            bool: True if the robot successfully moved to the specified positions, False otherwise.
        """
        print("move_to_joint_positions")
        curr_joints = self._robot.get_joint_state()
        
        # Check if current joints and positions are same
        assert len(list(positions)) == len(curr_joints), "positions and current joint angles are not same"


        # print("curr_joints", curr_joints)
        # print("len(list(positions))", len(list(positions)), "len(curr_joints)", len(curr_joints))
        if len(list(positions)) == len(curr_joints):
            max_delta = (np.abs(curr_joints - positions)).max()
            # print("max_delta", max_delta)   
            steps = min(int(max_delta / 0.01), 100)
            print("Steps to reach home pose", steps)

        if not delta:
            for jnt in np.linspace(curr_joints, positions, steps):
                # print("Commanding joint state")
                self._robot.command_joint_state(jnt)
                print(self._robot.get_ee_pose())
        elif delta:
            self._robot.command_joint_state(positions)

        return True

    def move_to_eef_positions(self, positions: np.ndarray, delta: bool = False):
        """Moves the robot to the specified joint positions.

        Args:
            positions (torch.Tensor): The joint positions to move the robot to.
            delta (bool, optional): Whether the positions are relative to the current positions. Defaults to False.

        Returns:
            bool: True if the robot successfully moved to the specified positions, False otherwise.
        """

        curr_pose = self._robot.get_ee_pose()
        assert len(positions) == len(curr_pose)
        if len(positions) == len(curr_pose):
            rpy_delta = np.abs(curr_pose[3:6] - positions[3:6])
            # ----- Extra Safety Check for RPY ----- # 
            for angle in rpy_delta:
                if abs(angle) > 0.75:
                    print(f"Angle: {angle}")
                    print(f"RPY input: {positions[3:6]}, RPY delta: {rpy_delta})")
                    raise ValueError("Angle difference is too large")
                
            max_delta = (np.abs(curr_pose - np.array(positions))).max()
            # print("max_delta", max_delta)   
            steps = min(int(max_delta / 0.001), 100)

            if not delta:
                for pose in np.linspace(curr_pose, positions, steps):
                    print("pose", pose)
                    self._robot.command_eef_pose(pose)
                    time.sleep(0.001)
            else:
                self._robot.command_eef_pose(positions)

        return True


    def get_action_space(self) -> tuple[list[float], list[float]]:
        if self.cfg.controller_type == "CARTESIAN_DELTA":
            high = [self.cfg.max_delta] * 3 + [self.cfg.max_delta * 3] * 3
            low = [-x for x in high]
        elif self.cfg.controller_type == "CARTESIAN_IMPEDANCE":
            low = self.ee_config.ee_range_low
            high = self.ee_config.ee_range_high
        else:
            raise ValueError("Invalid Controller type provided")

        # Add the gripper action space
        low.append(0.0)
        high.append(1.0)
        return low, high

    def update_gripper(self, gripper_action: float, blocking=False) -> None:
        # We always run the gripper in absolute position
        gripper_action = max(min(gripper_action, 1), 0)
        width = self._max_gripper_width * (1 - gripper_action)

        self.desired_gripper_qpos = gripper_action


    def update(self, action: list[float]) -> None:
        """
        Updates the robot controller with the action
        """
        assert len(action) == 7, f"wrong action dim: {len(action)}"
        # assert self._robot.is_running_policy(), "policy not running"

        '''
        # if not self._robot.is_running_policy():
        #     print("restarting cartesian impedance controller")
        #     self._robot.start_cartesian_impedance()
        #     time.sleep(1)
        '''
        # print(self.action_space.assert_in_range(action))
        # assert self.action_space.assert_in_range(action)
        if self.action_space.assert_in_range(action):
            pass
        else:
            raise ValueError("Action out of range")

        if self.use_gripper:
            robot_action: np.ndarray = np.array(action[:-1])
            gripper_action: float = action[-1]
        else:
            robot_action: np.ndarray = np.array(action)
            # gripper_action: float = 0


        if self.cfg.controller_type == "CARTESIAN_DELTA":
            pos = self._robot.get_ee_pose()[:-1]
            ee_pos, ee_ori = np.split(pos, [3])
            delta_pos, delta_ori = np.split(robot_action, [3])

            # compute new pos and new quat
            new_pos = ee_pos + delta_pos
            new_rot = ee_ori + delta_ori

            if self.use_gripper:
                end_eff_pos = np.concatenate((new_pos, new_rot, [gripper_action]))
            else:
                end_eff_pos = np.concatenate((new_pos, new_rot))

            # --- overwrite if the z exceeds ------ #
            if end_eff_pos[0] < self.x_min:
                end_eff_pos[0] = self.x_min
            if end_eff_pos[1] < self.y_min:
                end_eff_pos[1] = self.y_min
            if end_eff_pos[2] < self.z_min: 
                end_eff_pos[2] = self.z_min
                
            if end_eff_pos[0] > self.x_max:
                end_eff_pos[0] = self.x_max
            if end_eff_pos[1] > self.y_max:
                end_eff_pos[1] = self.y_max
            if end_eff_pos[2] > self.z_max:
                end_eff_pos[2] = self.z_max
            
            # --- Raise error if rpy absolute is more than threshold --- #
            # r, p, y = end_eff_pos[3:6]
            # if r > self.r_max or r < self.r_min:
            #     raise ValueError("Roll angle -- Out of bound")
            # if p > self.p_max or p < self.p_min:
            #     raise ValueError("Pitch angle -- Out of bound")
            # if y > self.y_max or y < self.y_min:
            #     raise ValueError("Yaw angle -- Out of bound")



            print(f"Abs Pose: {end_eff_pos}")

            self.move_to_eef_positions(end_eff_pos, delta=True)
            if self.use_gripper:
                self.update_gripper(gripper_action, blocking=False)
            # else:
                # print("Action Skipped due to out of range")
        else:
            raise ValueError("Invalid Controller type provided")



    def update_joint(self, action) -> None:
        """
        Updates the robot controller with the action
        """
        if self.use_gripper:
            assert len(action) == 7, f"wrong action dim: {len(action)}"
        else:
            assert len(action) == 6, f"wrong action dim: {len(action)}"
        

        # Check if the actions are in range
        joints_min = [-2.355, -2.355, -2.355, -2.355, -3.14, -3.14, 0]
        joints_max = [-0.785, -0.785, -0.785, -0.785, 3.14, 3.14, 0]

        # Check if the action is in range and print which joint is out of range
        if np.any(action < joints_min) or np.any(action > joints_max):
            print(f"Joint angles: {action}")
            raise ValueError("Joint angles out of range")


        # Get the change in joint angles
        curr_joints = self._robot.get_joint_state()
        assert(len(curr_joints) == len(action)), "Action and current joint angles are not same"

        delta = curr_joints - action
        #check if the delta is not more than 0.5 and print which joint is out of range
        if np.abs(delta).max() > 0.5:
            print(f"Delta: {delta}")
            print(f"Joint angles: {action}")
            raise ValueError("Joint angles difference is too large")
        
        self.move_to_joint_positions(action, delta=True)
        print("Joint angles updated . . . ")






    def reset(self, randomize: bool = False) -> None:
        print("reset env")

        if self.use_gripper:
            self.update_gripper(0)  # open the gripper

        # self.ee_config.reset(self, robot=self._robot)

        home = self.ee_config.home

        if randomize:
            # TODO: adjust this noise
            high = 0.01 * np.ones_like(home)
            noise = np.random.uniform(low=-high, high=high)
            print("home noise:", noise)
            home = home + noise
            self._robot.set_home_pose(home)

        self.go_home(blocking=False)

        # self.reached_place=False
        # self.reached_z_min = False
        # time.sleep(1)

    def get_state(self) -> dict[str, list[float]]:
        """
        Returns the robot state dictionary.
        For VR support MUST include [ee_pos, ee_quat]
        """
        ee_pos, ee_quat = np.split(self._robot.get_ee_pose()[:-1], [3])
        if self.use_gripper:
            gripper_state = self._gripper.get_current_position()
            # gripper_pos = 1 - (gripper_state / self._max_gripper_width) # 0 is open and 1 is closed
            gripper_pos = (gripper_state / self._max_gripper_width) # 0 is open and 1 is closed
        else:
            gripper_state = None
            gripper_pos = None
        joint_states = self._robot.get_joint_state()

        state = {
            "robot0_eef_pos": list(ee_pos),
            "robot0_eef_quat": list(ee_quat),
            "robot0_gripper_qpos": [gripper_pos],
            "robot0_desired_gripper_qpos": [self.desired_gripper_qpos],
            "joint_positions": list(joint_states)
        }

        in_good_range = self.ee_config.ee_in_good_range(
            state["robot0_eef_pos"], state["robot0_eef_quat"], False
        )
        return state, in_good_range


import datetime
import glob




@dataclass
class Args:
    agent: str = "ur"
    hostname: str = "10.104.59.112"
    # hostname: str = "192.168.77.243"
    robot_port: int = 50003  # for trajectory
    robot_ip: str = "192.168.77.21" 
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_pose: Optional[Tuple[float, ...]] = None

    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    verbose: bool = False
    camera_clients = {
    # you can optionally add camera nodes here for imitation learning purposes
    # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
    # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
    }
    controller: URTDEControllerConfig = field(
        default_factory=lambda: URTDEControllerConfig()
    )

if __name__ == "__main__":
    args = Args()
    
    # cfg = pyrallis.parse(config_class=PolyMainConfig)  # type: ignore
    np.set_printoptions(precision=4, linewidth=100, suppress=True)

    controller = URTDEController(args.controller, args.controller.task)

    time.sleep(3)


    # # -------- Update the gripper -----------
    # controller.update_gripper(0.5)
    # time.sleep(3)

    # # --------- Reset the robot ------------
    # controller.reset(randomize=False)
