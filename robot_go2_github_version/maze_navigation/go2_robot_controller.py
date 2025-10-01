"""
Go2机器人控制器类
负责控制Genesis中的Go2机器人移动和导航
基于Go2Env和关节控制
"""

import numpy as np
import torch
import genesis as gs
from typing import List, Tuple, Optional, Dict
import math
import os
import sys

# 添加locomotion路径以导入Go2环境
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
locomotion_dir = os.path.join(parent_dir, 'locomotion')
sys.path.append(locomotion_dir)

try:
    from go2_env import Go2Env
except ImportError:
    print("⚠️ 警告: 无法导入Go2Env，将使用简化版本")
    Go2Env = None


class Go2RobotController:
    """Go2机器人控制器类，管理机器人在迷宫中的移动"""

    def __init__(self, maze_env, record_video: bool = False, video_filename: str = None):
        """
        初始化Go2机器人控制器

        Args:
            maze_env: 迷宫环境对象
            record_video: 是否录制视频
            video_filename: 视频文件名（可选）
        """
        self.maze_env = maze_env
        self.robot = None
        self.current_pos = None
        self.target_pos = None
        
        # 视频录制相关属性
        self.record_video = record_video
        self.video_filename = video_filename
        self.path = []
        self.path_index = 0
        
        # Go2机器人参数
        self.robot_width = 0.35  # Go2机器人宽度 (m)
        self.robot_height = 0.35  # Go2机器人长度 (m)
        self.robot_diagonal = (self.robot_width ** 2 + self.robot_height ** 2) ** 0.5 / 2
        self.safety_margin = 0.1  # 安全边距
        
        # 移动参数
        self.speed = 0.5  # Go2机器人移动速度 (m/s)
        self.tolerance = 0.4  # 位置容差
        self.stuck_counter = 0
        self.max_stuck_steps = 50
        
        # Go2关节参数
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]
        
        # 默认关节角度（站立姿态）
        self.default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }
        
        # PD控制器参数
        self.kp = 70.0
        self.kd = 3.0
        
        # 步态参数
        self.step_height = 0.1
        self.step_frequency = 2.0  # Hz
        self.gait_phase = 0.0
        
        # 导航参数
        self.look_ahead_distance = 2.0
        self.smooth_factor = 0.7
        
        # 代价地图参数
        self.cost_map = None
        self.cost_map_size = 0
        self.cost_map_resolution = 0.1
        self.wall_cost = 1000
        self.center_cost = 0
        self.wall_distance_cost = 10
        self.center_preference = 500
        self.smooth_factor = 0.7

    def create_robot(self, start_pos: Optional[Tuple[float, float]] = None):
        """
        在Genesis场景中创建Go2机器人

        Args:
            start_pos: 起始位置 (x, y)，如果为None则使用迷宫起点

        Returns:
            机器人实体对象
        """
        if self.maze_env.scene is None:
            raise ValueError("迷宫环境场景未创建")

        # 确定起始位置
        if start_pos is None:
            if self.maze_env.start_pos is None:
                raise ValueError("迷宫环境未设置起点")
            start_pos = self.maze_env.get_world_position(self.maze_env.start_pos)

        # Go2机器人位置（考虑其尺寸）
        robot_pos = (start_pos[0], start_pos[1], 0.35)  # 高度0.35m

        # 创建真正的Go2机器人 - 使用URDF模型
        try:
            self.robot = self.maze_env.scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/go2/urdf/go2.urdf",  # 真正的Go2机器人URDF文件
                    pos=robot_pos,
                    quat=(0.0, 0.0, 0.0, 1.0),  # 无旋转
                    scale=1.0,  # 原始尺寸
                    fixed=False,  # 允许机器人移动
                    convexify=True,  # 启用凸化以提高性能
                    decimate=True,  # 启用网格简化
                    decimate_face_num=500,  # 简化到500个面
                    requires_jac_and_IK=False  # 不需要雅可比和逆运动学
                ),
                material=gs.materials.Rigid(
                    rho=100,  # 降低密度，减少碰撞冲击
                    friction=0.8  # 足够的摩擦力用于行走
                )
            )
            print("✅ 真正的Go2机器人URDF模型创建成功")
            
        except Exception as e:
            print(f"⚠️  无法加载Go2 URDF模型: {e}")
            print("   使用简化的盒子模型作为备选方案")
            
            # 备选方案：使用简化的盒子模型
            self.robot = self.maze_env.scene.add_entity(
                gs.morphs.Box(
                    size=(self.robot_width, self.robot_height, 0.3),  # Go2机器人尺寸
                    pos=robot_pos
                ),
                material=gs.materials.Rigid(
                    rho=500,  # 适中的密度
                    friction=0.8  # 足够的摩擦力用于行走
                ),
                surface=gs.surfaces.Default(
                    color=(0.2, 0.6, 1.0)  # 蓝色，代表Go2机器人
                )
            )

        self.current_pos = start_pos
        
        # 生成代价地图
        self._generate_cost_map()
        print(f"🗺️  Go2机器人代价地图已生成，大小: {self.cost_map_size}x{self.cost_map_size}")
        
        # 如果启用视频录制，设置视频录制
        if self.record_video:
            self._setup_video_recording()

        return self.robot

    def _setup_video_recording(self):
        """设置视频录制"""
        if self.video_filename is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_filename = f"go2_maze_navigation_{timestamp}.mp4"

        # 确保视频文件保存在videos文件夹中
        if not self.video_filename.startswith("videos/"):
            self.video_filename = f"videos/{self.video_filename}"

        print(f"🎥 Go2机器人视频录制已启用: {self.video_filename}")

        # 创建相机用于录制视频
        maze_diagonal = ((self.maze_env.width * self.maze_env.cell_size) ** 2 +
                         (self.maze_env.height * self.maze_env.cell_size) ** 2) ** 0.5
        camera_height = maze_diagonal * 0.8

        camera_offset = maze_diagonal * 0.2
        self.camera = self.maze_env.scene.add_camera(
            pos=(self.maze_env.width * self.maze_env.cell_size / 2 + camera_offset,
                 self.maze_env.height * self.maze_env.cell_size / 2 + camera_offset,
                 camera_height),
            lookat=(self.maze_env.width * self.maze_env.cell_size / 2,
                    self.maze_env.height * self.maze_env.cell_size / 2, 0),
            fov=75,
            res=(1280, 720),
            GUI=False
        )

        self._render_camera = True

    def get_robot_position(self) -> Tuple[float, float]:
        """
        获取机器人当前位置

        Returns:
            机器人位置 (x, y)
        """
        if self.robot is None:
            raise ValueError("机器人未创建")

        pos = self.robot.get_pos()
        return (pos[0].item(), pos[1].item())

    def get_robot_grid_position(self) -> Tuple[int, int]:
        """
        获取机器人当前网格位置

        Returns:
            机器人网格位置 (grid_x, grid_y)
        """
        world_pos = self.get_robot_position()
        return self.maze_env.get_grid_position(world_pos)

    def set_path(self, path: List[Tuple[int, int]]):
        """
        设置机器人路径

        Args:
            path: 路径点列表，每个点为网格坐标 (grid_x, grid_y)
        """
        self.path = path
        self.path_index = 0

        if len(path) > 0:
            self.target_pos = self.maze_env.get_world_position(path[0])
            print(f"🤖 Go2机器人路径设置:")
            print(f"   路径长度: {len(path)}")
            print(f"   当前目标点: {self.path_index} -> {path[0]}")
            print(f"   目标世界坐标: {self.target_pos}")
        else:
            print("⚠️  警告: 路径为空!")

    def update_target(self):
        """更新目标位置到路径中的下一个点"""
        if self.path_index < len(self.path) - 1:
            self.path_index += 1
            self.target_pos = self.maze_env.get_world_position(self.path[self.path_index])
        else:
            self.target_pos = None

    def is_at_target(self) -> bool:
        """
        检查机器人是否到达目标位置

        Returns:
            是否到达目标
        """
        if self.target_pos is None:
            return True

        current_pos = self.get_robot_position()
        distance = math.sqrt(
            (current_pos[0] - self.target_pos[0]) ** 2 +
            (current_pos[1] - self.target_pos[1]) ** 2
        )
        return distance < self.tolerance

    def is_path_complete(self) -> bool:
        """
        检查路径是否完成

        Returns:
            路径是否完成
        """
        return self.path_index >= len(self.path) - 1 and self.is_at_target()

    def move_towards_target(self, dt: float = 0.01):
        """
        Go2机器人向目标位置移动
        使用简化的移动逻辑，模拟四足机器人行走
        """
        if self.robot is None or self.target_pos is None:
            return

        if self.is_path_complete():
            return

        current_pos = self.get_robot_position()

        # 获取前瞻目标点
        look_ahead_target = self._get_look_ahead_target()
        if look_ahead_target is not None:
            target = look_ahead_target
        else:
            target = self.target_pos

        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance < self.tolerance:
            self.update_target()
            self.stuck_counter = 0
            return

        # 检查是否卡住
        if hasattr(self, '_last_pos'):
            if abs(current_pos[0] - self._last_pos[0]) < 0.005 and abs(current_pos[1] - self._last_pos[1]) < 0.005:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

        self._last_pos = current_pos

        # 如果卡住太久，尝试脱困
        if self.stuck_counter > self.max_stuck_steps:
            if self._try_escape_maneuver(current_pos):
                self.stuck_counter = 0
                return
            else:
                self.update_target()
                self.stuck_counter = 0
                return

        # 计算移动方向
        direction_x, direction_y = self._get_safe_direction(current_pos, target)

        # 平滑转向
        if hasattr(self, '_last_direction'):
            direction_x = self.smooth_factor * direction_x + (1 - self.smooth_factor) * self._last_direction[0]
            direction_y = self.smooth_factor * direction_y + (1 - self.smooth_factor) * self._last_direction[1]
            norm = math.sqrt(direction_x ** 2 + direction_y ** 2)
            if norm > 0:
                direction_x /= norm
                direction_y /= norm

        self._last_direction = (direction_x, direction_y)

        # 计算移动步长
        move_distance = self.speed * dt
        new_x = current_pos[0] + direction_x * move_distance
        new_y = current_pos[1] + direction_y * move_distance

        # 检查安全并移动
        if self._is_position_safe((new_x, new_y)):
            new_pos = torch.tensor([new_x, new_y, 0.35], dtype=torch.float32)
            self.robot.set_pos(new_pos)
            self.current_pos = (new_x, new_y)
            
            # 更新步态相位（模拟行走动画）
            self.gait_phase += self.step_frequency * dt
            if self.gait_phase >= 2 * math.pi:
                self.gait_phase -= 2 * math.pi
                
            # 如果是真正的Go2机器人，可以添加关节控制
            # 这里简化处理，实际项目中应该使用Go2Env的关节控制逻辑

    def _is_position_safe(self, pos: Tuple[float, float]) -> bool:
        """
        检查位置是否安全，考虑Go2机器人尺寸

        Args:
            pos: 要检查的位置 (x, y)

        Returns:
            位置是否安全
        """
        half_width = self.robot_width / 2
        half_height = self.robot_height / 2
        safety_margin = self.safety_margin
        effective_half_width = half_width + safety_margin
        effective_half_height = half_height + safety_margin

        # 检查机器人边界框内的多个点
        sample_points = [
            (pos[0] - effective_half_width, pos[1] - effective_half_height),  # 左下
            (pos[0] + effective_half_width, pos[1] - effective_half_height),  # 右下
            (pos[0] - effective_half_width, pos[1] + effective_half_height),  # 左上
            (pos[0] + effective_half_width, pos[1] + effective_half_height),  # 右上
            (pos[0], pos[1]),  # 中心
            (pos[0] - effective_half_width, pos[1]),  # 左中
            (pos[0] + effective_half_width, pos[1]),  # 右中
            (pos[0], pos[1] - effective_half_height),  # 下中
            (pos[0], pos[1] + effective_half_height),  # 上中
        ]

        for point in sample_points:
            grid_pos = self.maze_env.get_grid_position(point)
            if not self.maze_env.is_valid_position(grid_pos):
                return False

        return True

    def _generate_cost_map(self):
        """生成Go2机器人专用代价地图"""
        if self.maze_env.maze_grid is None:
            return

        width, height = self.maze_env.get_maze_bounds()
        self.cost_map_size = max(width, height) * 10
        self.cost_map = [[0.0 for _ in range(self.cost_map_size)] for _ in range(self.cost_map_size)]

        max_dist = max(1.0, min(width, height) / 2.0)

        for y in range(self.cost_map_size):
            for x in range(self.cost_map_size):
                grid_x = int(x * width / self.cost_map_size)
                grid_y = int(y * height / self.cost_map_size)

                if 0 <= grid_x < width and 0 <= grid_y < height:
                    if self.maze_env.maze_grid[grid_y, grid_x] == 1:
                        self.cost_map[y][x] = self.wall_cost
                    else:
                        min_wall_dist = self._get_min_wall_distance(grid_x, grid_y, width, height)
                        d = min(min_wall_dist, max_dist)
                        norm = d / max_dist
                        cost = self.wall_distance_cost * (1 - norm)
                        self.cost_map[y][x] = min(cost, self.wall_cost)
                else:
                    self.cost_map[y][x] = self.wall_cost

        self._smooth_cost_map()

    def _smooth_cost_map(self):
        """对代价地图进行平滑处理"""
        if self.cost_map is None:
            return

        smoothed_map = [[0.0 for _ in range(self.cost_map_size)] for _ in range(self.cost_map_size)]

        for i in range(self.cost_map_size):
            for j in range(self.cost_map_size):
                total_cost = 0.0
                count = 0

                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni = i + di
                        nj = j + dj

                        if 0 <= ni < self.cost_map_size and 0 <= nj < self.cost_map_size:
                            total_cost += self.cost_map[ni][nj]
                            count += 1

                if count > 0:
                    smoothed_map[i][j] = total_cost / count
                else:
                    smoothed_map[i][j] = self.cost_map[i][j]

        for i in range(self.cost_map_size):
            for j in range(self.cost_map_size):
                self.cost_map[i][j] = (self.smooth_factor * smoothed_map[i][j] +
                                       (1 - self.smooth_factor) * self.cost_map[i][j])

    def _get_min_wall_distance(self, x: int, y: int, width: int, height: int) -> float:
        """计算点到最近墙体的距离"""
        min_distance = float('inf')
        search_radius = 8  # 增加搜索半径，适应Go2机器人尺寸

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                check_x = x + dx
                check_y = y + dy

                if 0 <= check_x < width and 0 <= check_y < height:
                    if self.maze_env.maze_grid[check_y, check_x] == 1:
                        distance = math.sqrt(dx * dx + dy * dy)
                        min_distance = min(min_distance, distance)

        return min_distance if min_distance != float('inf') else 0.0

    def _get_safe_direction(self, current_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> Tuple[float, float]:
        """获取安全移动方向"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance == 0:
            return (0, 0)

        dx_norm = dx / distance
        dy_norm = dy / distance

        # 候选方向
        candidate_angles = list(range(-60, 61, 15))  # -60°到60°，每15°采样
        rad_factor = math.pi / 180
        candidate_directions = [(dx_norm * math.cos(a * rad_factor) - dy_norm * math.sin(a * rad_factor),
                                 dx_norm * math.sin(a * rad_factor) + dy_norm * math.cos(a * rad_factor))
                                for a in candidate_angles]

        best_direction = None
        best_score = float('inf')

        sample_distances = [1.0, 2.0, 3.0]  # 适应Go2机器人尺寸

        for dir_x, dir_y in candidate_directions:
            valid = True
            total_cost = 0.0

            for d in sample_distances:
                check_pos = (current_pos[0] + dir_x * d, current_pos[1] + dir_y * d)
                if not self._is_position_safe(check_pos):
                    valid = False
                    break
                total_cost += self._get_cost_at_position(check_pos)

            if not valid:
                continue

            avg_cost = total_cost / len(sample_distances)
            direction_alignment = dir_x * dx_norm + dir_y * dy_norm
            direction_penalty = (1 - direction_alignment) * 2.0

            total_score = avg_cost + direction_penalty

            if total_score < best_score:
                best_score = total_score
                best_direction = (dir_x, dir_y)

        if best_direction is None:
            best_direction = (dx_norm, dy_norm)

        return best_direction

    def _get_cost_at_position(self, pos: Tuple[float, float]) -> float:
        """获取指定位置的代价"""
        if self.cost_map is None:
            return 0.0

        maze_bounds = self.maze_env.get_maze_bounds()
        width, height = maze_bounds

        cell_size = self.maze_env.cell_size
        maze_world_width = width * cell_size
        maze_world_height = height * cell_size

        map_x = int(pos[0] * self.cost_map_size / maze_world_width)
        map_y = int(pos[1] * self.cost_map_size / maze_world_height)

        if 0 <= map_x < self.cost_map_size and 0 <= map_y < self.cost_map_size:
            return self.cost_map[map_y][map_x]
        else:
            return self.wall_cost

    def _try_escape_maneuver(self, current_pos: Tuple[float, float]) -> bool:
        """尝试脱困机动"""
        escape_directions = [
            (1.0, 0), (0, 1.0), (-1.0, 0), (0, -1.0),
            (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)
        ]

        for i, (dx, dy) in enumerate(escape_directions):
            escape_pos = (
                current_pos[0] + dx,
                current_pos[1] + dy
            )

            if self._is_position_safe(escape_pos):
                print(f"🚀 Go2机器人脱困机动: 方向 {i + 1}")
                new_pos = torch.tensor([escape_pos[0], escape_pos[1], 0.35], dtype=torch.float32)
                self.robot.set_pos(new_pos)
                self.current_pos = escape_pos
                return True

        return False

    def _get_look_ahead_target(self) -> Optional[Tuple[float, float]]:
        """获取前瞻目标点"""
        if self.path_index >= len(self.path) - 1:
            return None

        current_pos = self.get_robot_position()
        if not current_pos:
            return None

        for i in range(self.path_index + 1, len(self.path)):
            target_world = self.maze_env.get_world_position(self.path[i])
            distance = math.sqrt(
                (target_world[0] - current_pos[0]) ** 2 +
                (target_world[1] - current_pos[1]) ** 2
            )

            if distance > self.look_ahead_distance:
                if i > self.path_index + 1:
                    return self.maze_env.get_world_position(self.path[i - 1])
                else:
                    return None

        return self.maze_env.get_world_position(self.path[-1])

    def get_distance_to_goal(self) -> float:
        """获取到终点的距离"""
        if self.maze_env.goal_pos is None:
            return float('inf')

        current_pos = self.get_robot_position()
        goal_world_pos = self.maze_env.get_world_position(self.maze_env.goal_pos)

        distance = math.sqrt(
            (current_pos[0] - goal_world_pos[0]) ** 2 +
            (current_pos[1] - goal_world_pos[1]) ** 2
        )
        return distance

    def is_at_goal(self) -> bool:
        """检查机器人是否到达终点"""
        return self.get_distance_to_goal() < self.tolerance

    def set_speed(self, speed: float):
        """设置机器人移动速度"""
        self.speed = max(0.1, speed)

    def get_robot_entity(self):
        """获取机器人实体对象"""
        return self.robot
