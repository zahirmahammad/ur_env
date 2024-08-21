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
    # robot_ip should be local because this runs on the nuc that connects to the robot
    # robot_ip_address: str = "localhost"
    controller_type: str = "CARTESIAN_DELTA"
    max_delta: float = 0.06
    mock: int = 0

    ## --------- Newly Added ----------- ##
    agent: str = "ur"
    hostname: str = "10.104.56.158"
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

    # # Define the bounds for the Franka Robot
    JOINT_LOW = np.array(
        [-2.6172, -1.832, -1.832, -1.832, 1.046, 1.046, 0], dtype=np.float32
    )
    JOINT_HIGH = np.array(
        [-0.5228, -1.308, -1.308, -1.308, 2.094, 2.094, 1], dtype=np.float32
    )

    def __init__(self, cfg: URTDEControllerConfig, task) -> None:
        print("URTDE Controller Initialized...")

        self.cfg = cfg
        assert self.cfg.controller_type in {
            "CARTESIAN_DELTA",
            "CARTESIAN_IMPEDANCE",
        }

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
            # self._robot = RobotEnv(robot_client, control_rate_hz=cfg.hz, camera_dict=cfg.camera_clients)
            self._robot = robot_client
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

        if hasattr(self._gripper, "_max_position") :
            # Should grab this from robotiq2f
            self._max_gripper_width = self._gripper._max_position
        else:
            self._max_gripper_width = 255  # default, from Robotiq Value

        self.desired_gripper_qpos = 0

        self.reached_place = False
        self.reached_z_min = False

    def hello(self):
        return "hello"


    # def get_observations(self) -> Dict[str, np.ndarray]:
    #     joints = self._robot.get_joint_state()
    #     pos_quat = self._robot.get_ee_pose()
    #     gripper_pos = np.array([joints[-1]])
    #     return {
    #         "joint_positions": joints,
    #         "joint_velocities": joints,
    #         "ee_pos_quat": pos_quat,
    #         "gripper_position": gripper_pos,
    #     }
    

    def go_home(self, blocking=True) -> None:
        """Calls move_to_joint_positions to the current home positions."""
        if blocking is False:
            assert (
                self.home_pose is not None
            ), "Home pose not assigned! Call 'set_home_pose(<joint_angles>)' to enable homing"
            return self.move_to_joint_positions(
                positions=self.home_pose, delta=False)



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
        # print("curr_joints", curr_joints)
        # print("len(list(positions))", len(list(positions)), "len(curr_joints)", len(curr_joints))
        if len(list(positions)) == len(curr_joints):
            max_delta = (np.abs(curr_joints - positions)).max()
            print("max_delta", max_delta)   
            steps = min(int(max_delta / 0.01), 100)
            print("Steps to reach home pose", steps)
        for jnt in np.linspace(curr_joints, positions, steps):
            # print("Commanding joint state")
            self._robot.command_joint_state(jnt)
            print(self._robot.get_ee_pose())
        return True

    def move_to_eef_positions(self, positions: np.ndarray, delta: bool = False):
        """Moves the robot to the specified joint positions.

        Args:
            positions (torch.Tensor): The joint positions to move the robot to.
            delta (bool, optional): Whether the positions are relative to the current positions. Defaults to False.

        Returns:
            bool: True if the robot successfully moved to the specified positions, False otherwise.
        """
        # print("move_to_eef_positions")
        # curr_pose = self._robot.get_observations()["ee_pos_quat"]
        # gripper_pos = self._robot.get_observations()["gripper_position"]

        curr_pose = self._robot.get_ee_pose()
        # print("curr_joints", curr_joints)
        # print("len(list(positions))", list(positions), "len(curr_joints)", curr_pose)
        # print("len(reset_pose)", len(positions), "len(curr_pose)", len(curr_pose))
        assert len(positions) == len(curr_pose)
        if len(positions) == len(curr_pose):
            rpy_delta = np.abs(curr_pose[3:6] - positions[3:6])   
            for angle in rpy_delta:
                if abs(angle) > 0.75:
                    print(f"Angle: {angle}")
                    print(f"RPY input: {positions[3:6]}, RPY delta: {rpy_delta})")

                    #### -- Fail Safe Disabled - temporarily --
                    # print("[Angle difference is too large] >/>/>/>/>/>/>/>/... Inverting Orientation")
                    # positions[3:6] = curr_pose[3:6]
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

    # def step(self, joints: np.ndarray) :
    #     """Step the environment forward.

    #     Args:
    #         joints: joint angles command to step the environment with.

    #     Returns:
    #         obs: observation from the environment.
    #     """
    #     assert len(joints) == (
    #         self.num_dofs()
    #     ), f"input:{len(joints)}, robot:{self.num_dofs()}"
    #     assert self.num_dofs() == len(joints)
    #     # self.command_eef_pose(joints)
    #     self.command_joint_state(joints)
    #     # self._rate.sleep()
    #     return self.get_observations()


    def get_action_space(self) -> tuple[list[float], list[float]]:
        if self.cfg.controller_type == "CARTESIAN_DELTA":
            high = [self.cfg.max_delta] * 3 + [self.cfg.max_delta * 4] * 3
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

        # self._gripper.move(
        #     position=width,
        #     speed=0.1,
        #     force=0.01,
        # )

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
            return

        # # Do not execute if elbow angle is almost horizontal
        # if self.get_observations()["joint_positions"][1] < -2.7:
        #     return

        
        robot_action: np.ndarray = np.array(action[:-1])
        gripper_action: float = action[-1]

        if self.cfg.controller_type == "CARTESIAN_DELTA":
            pos = self._robot.get_ee_pose()
            ee_pos, ee_ori = np.split(pos, [3])
            delta_pos, delta_ori = np.split(robot_action, [3])

            # compute new pos and new quat
            new_pos = ee_pos + delta_pos
            # TODO: this can be made much faster using purpose build methods instead of scipy.
            # new_rot = (delta_ori * ee_ori).astype(np.float32)
            # new_rot = ee_ori + delta_ori
            # new_rot = ee_ori
            
            new_rot = np.array([3.14, 0.00, 0.002])

            # clip
            # new_pos, new_rot = self.ee_config.clip(new_pos, new_rot)
            end_eff_pos = np.concatenate((new_pos, new_rot, [gripper_action]))

            # Check if in good range
            # in_good_range = self.ee_config.ee_in_good_range(end_eff_pos[:3], end_eff_pos[3:6])
            # if in_good_range:

            # --- skip if the z is touching table ------ #

            # ## ----- This temporrary -> put this back in predication and rl_hardware code ----- ##
            z_min = -0.015
            # z_min = 0.008
            # z_max=0.14
            # if end_eff_pos[2] <= z_min and self.reached_z_min == False:
            if end_eff_pos[2] <= z_min: 
            #     # end_eff_pos[0] = -5.30370915e-02
            #     # end_eff_pos[1] = -2.90102685e-01
            #     # end_eff_pos[1] = end_eff_pos[1] - 0.002
                end_eff_pos[2] = z_min
            #     # end_eff_pos[2] = 0.005
            #     # self._robot.move_to_eef_positions(end_eff_pos, delta=False)
            #     self.reached_z_min = True
            
            # if self.reached_z_min:
            #     end_eff_pos[-1] = 0.8
            # # add condition, if gripper pose is 0.8 and the z is greater than z_max, then z = z_max

            # # if self.reached_place:
            # #     end_eff_pos[-1] = gripper_action



            # if end_eff_pos[-1] >= 0.5 and end_eff_pos[2] >= 0.135 and self.reached_place == False:
            #     # end_eff_pos[2] = z_max
            #     end_eff_pos[0] = -0.21
            #     end_eff_pos[1] = -0.35
            #     self._robot.move_to_eef_positions(end_eff_pos, delta=False)
            #     end_eff_pos[-1] = gripper_action
            #     # end_eff_pos[-1] = 0.5
            #     self.reached_place=True



            # print("Action: ", end_eff_pos)

            print(f"Abs Pose: {end_eff_pos}")

            self.move_to_eef_positions(end_eff_pos, delta=True)
            self.update_gripper(end_eff_pos[-1], blocking=False)
            # else:
                # print("Action Skipped due to out of range")
        else:
            raise ValueError("Invalid Controller type provided")

        # Update the gripper - Already included in move_to_eef_positions
        # self.update_gripper(gripper_action, blocking=False)


    def reset(self, randomize: bool = False) -> None:
        print("reset env")

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

        self._robot.go_home(blocking=False)

        # assert not self._robot.is_running_policy()
        # self._robot.start_cartesian_impedance()
        self.reached_place=False
        self.reached_z_min = False
        time.sleep(1)

    def get_state(self) -> dict[str, list[float]]:
        """
        Returns the robot state dictionary.
        For VR support MUST include [ee_pos, ee_quat]
        """
        ee_pos, ee_quat = np.split(self._robot.get_ee_pose(), [3])
        gripper_state = self._gripper.get_current_position()
        joint_states = self._robot.get_joint_state()
        # gripper_pos = 1 - (gripper_state / self._max_gripper_width) # 0 is open and 1 is closed
        gripper_pos = (gripper_state / self._max_gripper_width) # 0 is open and 1 is closed

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
    hostname: str = "10.104.56.158"
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

    # robot = controller._robot

    # Print the current ee position
    # print(robot.get_ee_pose())

    # Go to a given ee position
    # # Reset Back
    # end_eff_pos_data = np.array([-5.30370915e-02, -2.90102685e-01,  5.24580323e-03,  3.14000000e+00,0.00000000e+00,  2.00000000e-03, 0.0])

    # print("Moving..")
    # robot.move_to_eef_positions(end_eff_pos_data, delta=False)

    # time.sleep(3)

    # controller.update_gripper(0.5)

    # time.sleep(3)

    # controller.reset(randomize=False)

    # # Execute the actions from hdf5
    # import h5py

    # with h5py.File("release/data/real_robot/data_processed_spaced.hdf5", "r") as f:
    #     # initial_pose = f["data"]['demo_1']['states'][0]
    #     actions = f["data"]["demo_0"]["actions"][:]
    
    # print("Executing actions...")
    # for action in actions:
    #     controller.update(action)
    #     time.sleep(0.5)
