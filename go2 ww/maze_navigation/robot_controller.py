"""
机器人控制器类
负责控制Genesis中的机器人移动和导航
"""

import numpy as np
import torch
import genesis as gs
from typing import List, Tuple, Optional, Dict
import math


class RobotController:
    """机器人控制器类，管理机器人在迷宫中的移动"""

    def __init__(self, maze_env, robot_type: str = "box", record_video: bool = False, video_filename: str = None,
                 urdf_file: str = None):
        """
        初始化机器人控制器

        Args:
            maze_env: 迷宫环境对象
            robot_type: 机器人类型 ("box", "sphere", "cylinder", "urdf")
            record_video: 是否录制视频
            video_filename: 视频文件名（可选）
            urdf_file: URDF文件路径（当robot_type为"urdf"时使用）
        """
        self.maze_env = maze_env
        self.robot_type = robot_type
        self.urdf_file = urdf_file
        self.robot = None
        self.current_pos = None
        self.target_pos = None

        # 视频录制相关属性
        self.record_video = record_video
        self.video_filename = video_filename
        self.path = []
        self.path_index = 0
        self.speed = 1.0  # 进一步降低移动速度，在复杂迷宫中更稳定
        self.rotation_speed = 2.0  # 提高旋转速度，更快转弯
        self.tolerance = 0.25  # 进一步增加位置容差，更容易到达目标点
        self.stuck_counter = 0  # 卡住计数器
        self.max_stuck_steps = 40  # 增加最大卡住步数，减少误判
        self.robot_radius = 0.08  # 进一步减小机器人碰撞半径，更容易通过拐角
        self.smooth_factor = 0.5  # 进一步降低转弯平滑因子，更直接的转弯
        self.look_ahead_distance = 1.2  # 进一步减少前瞻距离，更保守的路径跟踪

        # 机器人尺寸参数
        self.robot_width = 0.25  # 进一步减小机器人宽度，更容易通过
        self.robot_height = 0.25  # 进一步减小机器人高度，更容易通过
        self.robot_diagonal = (self.robot_width ** 2 + self.robot_height ** 2) ** 0.5 / 2  # 对角线的一半
        self.safety_margin = 0.05  # 进一步减小安全边距，更宽松

        # 局部路径规划参数
        self.local_planning = True  # 打开局部路径规划，简化移动逻辑
        self.avoidance_distance = max(self.robot_diagonal + 0.1, 0.8)  # 避障距离

        # 代价地图参数
        self.cost_map = None  # 代价地图
        self.cost_map_size = 0  # 代价地图大小
        self.cost_map_resolution = 0.1  # 代价地图分辨率
        self.wall_cost = 1000  # 墙体代价
        self.center_cost = 0  # 道路中央代价
        self.wall_distance_cost = 10  # 距离墙体代价系数（进一步降低）
        self.center_preference = 500 # 道路中央偏好系数（大幅增加）
        self.smooth_factor = 0.7  # 代价平滑因子（增加平滑度）

        # 脱困策略参数
        self.escape_attempts = 0  # 脱困尝试次数
        self.max_escape_attempts = 4  # 最大脱困尝试次数
        self.escape_distance = 0.6  # 脱困移动距离

    def create_robot(self, start_pos: Optional[Tuple[float, float]] = None):
        """
        在Genesis场景中创建机器人

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

        # 创建机器人 - 确保在开放区域内，避免与墙体重叠
        # 问题分析：机器人位置与相邻墙体重叠，需要调整位置
        # 将机器人位置向开放区域中心偏移更多
        robot_pos = (start_pos[0] - 0.3, start_pos[1] - 0.3, 0.3)  # 向开放区域中心偏移更多

        if self.robot_type == "box":
            self.robot = self.maze_env.scene.add_entity(
                gs.morphs.Box(
                    size=(0.36, 0.36, 0.3),  # 缩小为60%：0.6*0.6=0.36, 0.5*0.6=0.3
                    pos=robot_pos
                ),
                material=gs.materials.Rigid(
                    rho=100,  # 降低密度，减少碰撞冲击
                    friction=0.1  # 降低摩擦力，让机器人能够移动
                ),
                surface=gs.surfaces.Default(
                    color=(1.0, 0.0, 0.0)  # 红色，更显著
                )
            )
        elif self.robot_type == "sphere":
            self.robot = self.maze_env.scene.add_entity(
                gs.morphs.Sphere(
                    radius=0.18,  # 缩小为60%：0.3*0.6=0.18
                    pos=robot_pos
                ),
                material=gs.materials.Rigid(
                    rho=100,  # 降低密度，减少碰撞冲击
                    friction=0.1  # 降低摩擦力，让机器人能够移动
                ),
                surface=gs.surfaces.Default(
                    color=(0.0, 1.0, 0.0)  # 绿色，更显著
                )
            )
        elif self.robot_type == "cylinder":
            self.robot = self.maze_env.scene.add_entity(
                gs.morphs.Cylinder(
                    radius=0.18,  # 缩小为60%：0.3*0.6=0.18
                    height=0.3,  # 缩小为60%：0.5*0.6=0.3
                    pos=robot_pos
                ),
                material=gs.materials.Rigid(
                    rho=100,  # 降低密度，减少碰撞冲击
                    friction=0.1  # 降低摩擦力，让机器人能够移动
                ),
                surface=gs.surfaces.Default(
                    color=(0.0, 0.0, 1.0)  # 蓝色，更显著
                )
            )
        elif self.robot_type == "urdf":
            if self.urdf_file is None:
                raise ValueError("使用URDF机器人类型时必须提供urdf_file参数")

            # 使用URDF文件创建机器人
            self.robot = self.maze_env.scene.add_entity(
                gs.morphs.URDF(
                    file=self.urdf_file,
                    pos=robot_pos,
                    scale=0.5,  # 可以根据需要调整缩放
                    fixed=True,  # 允许机器人移动
                    convexify=True,  # 启用凸化以提高性能
                    decimate=True,  # 启用网格简化
                    decimate_face_num=500,  # 简化到500个面
                    requires_jac_and_IK=False  # 不需要雅可比和逆运动学
                ),
                material=gs.materials.Rigid(
                    rho=200,  # 降低密度，减少碰撞冲击
                    friction=0.1  # 降低摩擦力，让机器人能够移动
                ),
                surface=gs.surfaces.Default(
                    color=(1.0, 0.5, 0.0)  # 橙色，区分URDF机器人
                )
            )
        else:
            raise ValueError(f"不支持的机器人类型: {self.robot_type}")

        self.current_pos = start_pos

        # 打开代价地图生成
        self._generate_cost_map()
        print(f"🗺️  代价地图已生成，大小: {self.cost_map_size}x{self.cost_map_size}")

        # 如果启用视频录制，设置视频录制
        if self.record_video:
            self._setup_video_recording()

        return self.robot

    def _setup_video_recording(self):
        """设置视频录制"""
        if self.video_filename is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_filename = f"maze_navigation_{timestamp}.mp4"

        # 确保视频文件保存在videos文件夹中
        if not self.video_filename.startswith("videos/"):
            self.video_filename = f"videos/{self.video_filename}"

        print(f"🎥 视频录制已启用: {self.video_filename}")

        # 创建相机用于录制视频 - 调整到更高的位置和更广的视角
        # 计算迷宫的对角线长度，确保相机能看到整个迷宫
        maze_diagonal = ((self.maze_env.width * self.maze_env.cell_size) ** 2 +
                         (self.maze_env.height * self.maze_env.cell_size) ** 2) ** 0.5
        camera_height = maze_diagonal * 20  # 相机高度为迷宫对角线的0.8倍

        # 将相机稍微偏移，获得更好的俯视角度
        camera_offset = maze_diagonal * 0.2
        self.camera = self.maze_env.scene.add_camera(
            pos=(self.maze_env.width * self.maze_env.cell_size / 2 + camera_offset,
                 self.maze_env.height * self.maze_env.cell_size / 2 + camera_offset,
                 camera_height),  # 更高的相机位置
            lookat=(self.maze_env.width * self.maze_env.cell_size / 2,
                    self.maze_env.height * self.maze_env.cell_size / 2, 0),
            fov=75,  # 进一步增加视野角度到75度
            res=(1280, 720),
            GUI=False
        )

        # 在导航循环中渲染相机图像
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
            # 将第一个路径点转换为世界坐标作为目标
            self.target_pos = self.maze_env.get_world_position(path[0])
            print(f"🤖 机器人路径设置:")
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
            # 已经到达路径末尾，设置为None表示完成
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
        向目标位置移动（改进版）
        使用前瞻目标、代价地图和局部避障结合
        """
        if self.robot is None or self.target_pos is None:
            return

        # 检查路径是否已经完成
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
            # 到达当前目标，更新下一个目标
            self.update_target()
            self.stuck_counter = 0
            return

        # 检查是否卡住
        if hasattr(self, '_last_pos'):
            if abs(current_pos[0] - self._last_pos[0]) < 0.003 and abs(current_pos[1] - self._last_pos[1]) < 0.003:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

        self._last_pos = current_pos

        # 如果卡住太久，尝试脱困
        if self.stuck_counter > self.max_stuck_steps:
            if self._try_escape_maneuver(current_pos):
                self.stuck_counter = 0
                self.escape_attempts = 0
                return
            else:
                # 跳过当前目标点
                self.update_target()
                self.stuck_counter = 0
                self.escape_attempts = 0
                return

        # 计算移动方向
        direction_x, direction_y = self._get_local_avoidance_direction(current_pos, target)

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

        # 检查安全
        if self._is_position_safe((new_x, new_y)):
            new_pos = torch.tensor([new_x, new_y, 0.1], dtype=torch.float32)
            self.robot.set_pos(new_pos)
            self.current_pos = (new_x, new_y)
        else:
            # 尝试更小步长
            smaller_move_distance = move_distance * 0.5
            new_x_small = current_pos[0] + direction_x * smaller_move_distance
            new_y_small = current_pos[1] + direction_y * smaller_move_distance
            if self._is_position_safe((new_x_small, new_y_small)):
                new_pos = torch.tensor([new_x_small, new_y_small, 0.1], dtype=torch.float32)
                self.robot.set_pos(new_pos)
                self.current_pos = (new_x_small, new_y_small)
            else:
                # 脱困机动
                self._try_escape_maneuver(current_pos)

    def move_to_position(self, target_world_pos: Tuple[float, float], dt: float = 0.01):
        """
        移动到指定世界坐标位置

        Args:
            target_world_pos: 目标世界坐标 (x, y)
            dt: 时间步长
        """
        if self.robot is None:
            return

        current_pos = self.get_robot_position()

        # 计算方向向量
        dx = target_world_pos[0] - current_pos[0]
        dy = target_world_pos[1] - current_pos[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance < self.tolerance:
            return

        # 计算移动方向
        if distance > 0:
            direction_x = dx / distance
            direction_y = dy / distance

            # 计算新位置
            move_distance = self.speed * dt
            new_x = current_pos[0] + direction_x * move_distance
            new_y = current_pos[1] + direction_y * move_distance

            # 检查新位置是否有效
            new_grid_pos = self.maze_env.get_grid_position((new_x, new_y))
            if self.maze_env.is_valid_position(new_grid_pos):
                # 更新机器人位置
                new_pos = torch.tensor([new_x, new_y, 0.1], dtype=torch.float32)
                self.robot.set_pos(new_pos)
                self.current_pos = (new_x, new_y)

    def reset_to_start(self):
        """重置机器人到起点"""
        if self.robot is None or self.maze_env.start_pos is None:
            return

        start_world_pos = self.maze_env.get_world_position(self.maze_env.start_pos)
        start_pos = torch.tensor([start_world_pos[0], start_world_pos[1], 0.1], dtype=torch.float32)

        self.robot.set_pos(start_pos)
        self.current_pos = (start_world_pos[0], start_world_pos[1])
        self.path = []
        self.path_index = 0
        self.target_pos = None

    def get_distance_to_goal(self) -> float:
        """
        获取到终点的距离

        Returns:
            到终点的距离
        """
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
        """
        检查机器人是否到达终点

        Returns:
            是否到达终点
        """
        return self.get_distance_to_goal() < self.tolerance

    def _is_position_safe(self, pos: Tuple[float, float]) -> bool:
        """
        检查位置是否安全，考虑机器人实际尺寸和占用空间

        Args:
            pos: 要检查的位置 (x, y)

        Returns:
            位置是否安全
        """
        # 使用更密集的采样点检查机器人占用空间
        half_width = self.robot_width / 2
        half_height = self.robot_height / 2

        # 增加安全边距
        safety_margin = self.safety_margin
        effective_half_width = half_width + safety_margin
        effective_half_height = half_height + safety_margin

        # 检查机器人边界框内的多个点
        # 使用更密集的网格采样
        sample_points = []

        # 边界点
        sample_points.extend([
            (pos[0] - effective_half_width, pos[1] - effective_half_height),  # 左下
            (pos[0] + effective_half_width, pos[1] - effective_half_height),  # 右下
            (pos[0] - effective_half_width, pos[1] + effective_half_height),  # 左上
            (pos[0] + effective_half_width, pos[1] + effective_half_height),  # 右上
        ])

        # 中心点
        sample_points.append((pos[0], pos[1]))

        # 边缘中点
        sample_points.extend([
            (pos[0] - effective_half_width, pos[1]),  # 左中
            (pos[0] + effective_half_width, pos[1]),  # 右中
            (pos[0], pos[1] - effective_half_height),  # 下中
            (pos[0], pos[1] + effective_half_height),  # 上中
        ])

        # 检查所有采样点
        for point in sample_points:
            grid_pos = self.maze_env.get_grid_position(point)
            if not self.maze_env.is_valid_position(grid_pos):
                return False

        return True

    def _generate_cost_map(self):
        """
        生成改进版代价地图，使机器人倾向于走道路中央并远离墙体
        """
        if self.maze_env.maze_grid is None:
            return

        width, height = self.maze_env.get_maze_bounds()
        self.cost_map_size = max(width, height) * 10  # 分辨率提升
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
                        norm = d / max_dist  # 0靠墙，1远离墙

                        # 中间最小代价，靠墙代价高
                        cost = self.wall_distance_cost * (1 - norm)
                        self.cost_map[y][x] = min(cost, self.wall_cost)
                else:
                    self.cost_map[y][x] = self.wall_cost

        # 平滑代价地图
        self._smooth_cost_map()

    def _smooth_cost_map(self):
        """
        对代价地图进行平滑处理，减少震荡
        """
        if self.cost_map is None:
            return

        # 创建平滑后的代价地图
        smoothed_map = [[0.0 for _ in range(self.cost_map_size)] for _ in range(self.cost_map_size)]

        # 对每个点进行平滑处理
        for i in range(self.cost_map_size):
            for j in range(self.cost_map_size):
                # 计算周围点的平均值
                total_cost = 0.0
                count = 0

                # 检查周围3x3区域
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

        # 使用平滑因子混合原始和平滑后的代价
        for i in range(self.cost_map_size):
            for j in range(self.cost_map_size):
                self.cost_map[i][j] = (self.smooth_factor * smoothed_map[i][j] +
                                       (1 - self.smooth_factor) * self.cost_map[i][j])

    def _get_min_wall_distance(self, x: int, y: int, width: int, height: int) -> float:
        """
        计算点到最近墙体的距离
        """
        min_distance = float('inf')

        # 搜索周围区域
        search_radius = 5
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                check_x = x + dx
                check_y = y + dy

                # 检查边界
                if 0 <= check_x < width and 0 <= check_y < height:
                    if self.maze_env.maze_grid[check_y, check_x] == 1:
                        distance = math.sqrt(dx * dx + dy * dy)
                        min_distance = min(min_distance, distance)

        return min_distance if min_distance != float('inf') else 0.0

    def _get_cost_at_position(self, pos: Tuple[float, float]) -> float:
        """
        获取指定位置的代价
        """
        if self.cost_map is None:
            return 0.0

        # 将世界坐标转换为代价地图坐标
        # 考虑迷宫的实际偏移和尺寸
        maze_bounds = self.maze_env.get_maze_bounds()
        width, height = maze_bounds

        # 计算迷宫的实际世界坐标范围
        cell_size = self.maze_env.cell_size
        maze_world_width = width * cell_size
        maze_world_height = height * cell_size

        # 将世界坐标映射到代价地图坐标
        # 假设迷宫从(0,0)开始，但需要考虑实际偏移
        map_x = int(pos[0] * self.cost_map_size / maze_world_width)
        map_y = int(pos[1] * self.cost_map_size / maze_world_height)

        # 检查边界
        if 0 <= map_x < self.cost_map_size and 0 <= map_y < self.cost_map_size:
            return self.cost_map[map_y][map_x]
        else:
            return self.wall_cost

    def _get_local_avoidance_direction(self, current_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> \
    Tuple[float, float]:
        """
        局部避障路径规划，使用代价地图选择最佳方向（多点采样+中央偏好+平滑）
        """
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance == 0:
            return (0, 0)

        dx_norm = dx / distance
        dy_norm = dy / distance

        # 候选方向
        candidate_angles = list(range(-90, 91, 10))  # -90°到90°，每10°采样
        rad_factor = math.pi / 180
        candidate_directions = [(dx_norm * math.cos(a * rad_factor) - dy_norm * math.sin(a * rad_factor),
                                 dx_norm * math.sin(a * rad_factor) + dy_norm * math.cos(a * rad_factor))
                                for a in candidate_angles]

        best_direction = None
        best_score = float('inf')

        # 前方多点采样距离
        sample_distances = [0.5, 1.0, 1.5]  # 米，可根据迷宫尺寸调整

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

            # 方向对齐惩罚
            direction_alignment = dir_x * dx_norm + dir_y * dy_norm
            direction_penalty = (1 - direction_alignment) * 1.5  # 减小方向惩罚

            # 稳定性
            stability_bonus = 0
            if hasattr(self, '_last_direction') and self._last_direction != (0, 0):
                last_alignment = dir_x * self._last_direction[0] + dir_y * self._last_direction[1]
                stability_bonus = last_alignment * 5  # 平滑方向权重

            total_score = avg_cost + direction_penalty - stability_bonus

            if total_score < best_score:
                best_score = total_score
                best_direction = (dir_x, dir_y)

        # 如果没有可行方向，直接朝目标
        if best_direction is None:
            best_direction = (dx_norm, dy_norm)

        # 平滑方向变化
        if hasattr(self, '_last_direction'):
            dir_x, dir_y = best_direction
            dir_x = self.smooth_factor * dir_x + (1 - self.smooth_factor) * self._last_direction[0]
            dir_y = self.smooth_factor * dir_y + (1 - self.smooth_factor) * self._last_direction[1]
            norm = math.sqrt(dir_x ** 2 + dir_y ** 2)
            if norm > 0:
                dir_x /= norm
                dir_y /= norm
            best_direction = (dir_x, dir_y)

        self._last_direction = best_direction
        return best_direction

    def _try_escape_maneuver(self, current_pos: Tuple[float, float]) -> bool:
        """
        尝试脱困机动 - 改进版本

        Args:
            current_pos: 当前位置 (x, y)

        Returns:
            是否成功脱困
        """
        if self.escape_attempts >= self.max_escape_attempts:
            return False

        self.escape_attempts += 1

        # 尝试多个方向的脱困移动，包括对角线方向
        escape_directions = [
            (0.5, 0),  # 右
            (0, 0.5),  # 上
            (-0.5, 0),  # 左
            (0, -0.5),  # 下
            (0.3, 0.3),  # 右上
            (-0.3, 0.3),  # 左上
            (0.3, -0.3),  # 右下
            (-0.3, -0.3),  # 左下
        ]

        for i, (dx, dy) in enumerate(escape_directions):
            escape_pos = (
                current_pos[0] + dx * self.escape_distance,
                current_pos[1] + dy * self.escape_distance
            )

            if self._is_position_safe(escape_pos):
                print(f"🚀 尝试脱困机动 {self.escape_attempts}: 方向 {i + 1}")
                # 直接移动到脱困位置
                new_pos = torch.tensor([escape_pos[0], escape_pos[1], 0.1], dtype=torch.float32)
                self.robot.set_pos(new_pos)
                self.current_pos = escape_pos
                return True

        return False

    def _get_look_ahead_target(self) -> Optional[Tuple[float, float]]:
        """
        获取前瞻目标点，用于更平滑的路径跟踪

        Returns:
            前瞻目标点的世界坐标，如果没有则返回None
        """
        if self.path_index >= len(self.path) - 1:
            return None

        current_pos = self.get_robot_position()
        if not current_pos:
            return None

        # 寻找前瞻距离内的最远可达点
        for i in range(self.path_index + 1, len(self.path)):
            target_world = self.maze_env.get_world_position(self.path[i])
            distance = math.sqrt(
                (target_world[0] - current_pos[0]) ** 2 +
                (target_world[1] - current_pos[1]) ** 2
            )

            # 如果距离超过前瞻距离，返回前一个点
            if distance > self.look_ahead_distance:
                if i > self.path_index + 1:
                    return self.maze_env.get_world_position(self.path[i - 1])
                else:
                    return None

        # 如果所有点都在前瞻距离内，返回最后一个点
        return self.maze_env.get_world_position(self.path[-1])

    def set_speed(self, speed: float):
        """
        设置机器人移动速度

        Args:
            speed: 移动速度 (m/s)
        """
        self.speed = max(0.1, speed)

    def get_robot_entity(self):
        """
        获取机器人实体对象

        Returns:
            机器人实体
        """
        return self.robot
