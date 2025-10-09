"""
迷宫环境类
负责创建和管理Genesis中的迷宫环境，包括墙体、起点、终点等
"""

import numpy as np
import genesis as gs
from typing import List, Tuple, Optional, Dict
import random


class MazeEnvironment:
    """迷宫环境类，管理Genesis场景中的迷宫布局"""

    def __init__(self, width: int = 10, height: int = 10, cell_size: float = 1.0):
        """
        初始化迷宫环境

        Args:
            width: 迷宫宽度（格子数）
            height: 迷宫高度（格子数）
            cell_size: 每个格子的大小（米）
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.maze_grid = None
        self.walls = []
        self.start_pos = None
        self.goal_pos = None
        self.scene = None
        self.video_filename = None

    def generate_maze(self, algorithm: str = "open") -> np.ndarray:
        """
        生成迷宫网格

        Args:
            algorithm: 迷宫生成算法 ("recursive_backtracking", "random", "simple", "open")

        Returns:
            迷宫网格，0表示通路，1表示墙壁
        """
        if algorithm == "recursive_backtracking":
            self.maze_grid = self._generate_recursive_backtracking()
        elif algorithm == "random":
            self.maze_grid = self._generate_random_maze()
        elif algorithm == "simple":
            self.maze_grid = self._generate_simple_maze()
        elif algorithm == "open":
            self.maze_grid = self._generate_open_maze()
        else:
            raise ValueError(f"未知的迷宫生成算法: {algorithm}")

        return self.maze_grid

    def _generate_recursive_backtracking(self) -> np.ndarray:
        """使用递归回溯算法生成迷宫 - 优化版本，确保复杂性和可达性"""
        # 创建全墙网格
        maze = np.ones((self.height * 2 + 1, self.width * 2 + 1), dtype=int)

        def carve_path(x, y):
            """递归挖掘路径"""
            maze[y, x] = 0

            # 随机打乱方向
            directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < maze.shape[1] and 0 <= ny < maze.shape[0] and
                        maze[ny, nx] == 1):
                    # 打通墙壁
                    maze[y + dy // 2, x + dx // 2] = 0
                    carve_path(nx, ny)

        # 从奇数位置开始挖掘
        start_x, start_y = 1, 1
        carve_path(start_x, start_y)

        # 设置起点和终点
        self.start_pos = (1, 1)
        self.goal_pos = (maze.shape[1] - 2, maze.shape[0] - 2)

        # 确保起点和终点是通路
        maze[self.start_pos[1], self.start_pos[0]] = 0
        maze[self.goal_pos[1], self.goal_pos[0]] = 0

        # 确保起点周围有至少一个出口（但保持迷宫复杂性）
        # 只确保起点不会被完全封闭，但允许复杂的路径
        start_exits = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = self.start_pos[0] + dx, self.start_pos[1] + dy
            if (0 <= nx < maze.shape[1] and 0 <= ny < maze.shape[0] and maze[ny, nx] == 0):
                start_exits += 1

        # 如果起点没有出口，只打通一个方向（保持迷宫复杂性）
        if start_exits == 0:
            # 优先向右或向下打通
            if self.start_pos[0] + 1 < maze.shape[1]:
                maze[self.start_pos[1], self.start_pos[0] + 1] = 0
            elif self.start_pos[1] + 1 < maze.shape[0]:
                maze[self.start_pos[1] + 1, self.start_pos[0]] = 0

        return maze

    def _generate_random_maze(self) -> np.ndarray:
        """生成随机迷宫"""
        maze = np.ones((self.height, self.width), dtype=int)

        # 随机生成一些通路
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                if random.random() < 0.3:  # 30%概率生成通路
                    maze[i, j] = 0

        # 确保起点和终点是通路
        self.start_pos = (0, 0)
        self.goal_pos = (self.width - 1, self.height - 1)
        maze[self.start_pos[1], self.start_pos[0]] = 0
        maze[self.goal_pos[1], self.goal_pos[0]] = 0

        return maze

    def _generate_simple_maze(self) -> np.ndarray:
        """生成简单迷宫（用于测试）"""
        maze = np.zeros((self.height, self.width), dtype=int)

        # 添加一些简单的墙壁
        if self.width > 4 and self.height > 4:
            # 中间添加一堵墙
            wall_x = self.width // 2
            for y in range(1, self.height - 1):
                maze[y, wall_x] = 1
            # 留一个缺口
            maze[self.height // 2, wall_x] = 0

        self.start_pos = (0, 0)
        self.goal_pos = (self.width - 1, self.height - 1)

        return maze

    def _generate_open_maze(self) -> np.ndarray:
        """生成开放迷宫 - 确保起点和终点之间有明确路径"""
        maze = np.zeros((self.height, self.width), dtype=int)

        # 设置起点和终点
        self.start_pos = (0, 0)
        self.goal_pos = (self.width - 1, self.height - 1)

        # 确保起点和终点是通路
        maze[self.start_pos[1], self.start_pos[0]] = 0
        maze[self.goal_pos[1], self.goal_pos[0]] = 0

        # 创建一条从起点到终点的基本路径
        # 先向右走到边界
        for x in range(self.start_pos[0], self.width - 1):
            maze[self.start_pos[1], x] = 0

        # 再向下走到终点
        for y in range(self.start_pos[1], self.height):
            maze[y, self.width - 1] = 0

        # 添加一些随机障碍物，但确保不阻塞主路径
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                # 避免在起点和终点附近添加障碍物
                if (abs(i - self.start_pos[1]) > 1 or abs(j - self.start_pos[0]) > 1) and \
                        (abs(i - self.goal_pos[1]) > 1 or abs(j - self.goal_pos[0]) > 1):
                    if random.random() < 0.2:  # 20%概率添加障碍物
                        maze[i, j] = 1

        return maze

    def create_genesis_scene(self, show_viewer: bool = True, record_video: bool = False,
                             video_filename: str = None) -> gs.Scene:
        """
        在Genesis中创建迷宫场景

        Args:
            show_viewer: 是否显示可视化窗口
            record_video: 是否录制视频
            video_filename: 视频文件名（如果为None则自动生成）

        Returns:
            Genesis场景对象
        """
        if self.maze_grid is None:
            raise ValueError("请先生成迷宫网格")

        # 初始化Genesis
        gs.init(backend=gs.cpu)

        # 创建场景
        viewer_options = gs.options.ViewerOptions(
            camera_pos=(self.width * self.cell_size / 2,
                        self.height * self.cell_size / 2,
                        max(self.width, self.height) * self.cell_size),
            camera_lookat=(self.width * self.cell_size / 2,
                           self.height * self.cell_size / 2, 0),
            camera_fov=45,
            max_FPS=60,
        )

        # 如果录制视频，设置视频录制选项
        if record_video:
            if video_filename is None:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"maze_navigation_{timestamp}.mp4"

            # 确保视频文件保存在videos文件夹中
            if not video_filename.startswith("videos/"):
                video_filename = f"videos/{video_filename}"

            print(f"🎥 视频录制已启用: {video_filename}")
            # 存储视频文件名供后续使用
            self.video_filename = video_filename

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                gravity=(0, 0, -9.81),
            ),
            viewer_options=viewer_options,
            show_viewer=show_viewer,
        )

        # 添加地面 - 确保完全覆盖迷宫
        # 根据实际迷宫网格尺寸计算地面大小
        if hasattr(self, 'maze_grid') and self.maze_grid is not None:
            # 递归回溯算法生成的迷宫尺寸是 (height * 2 + 1, width * 2 + 1)
            # 计算实际迷宫的世界坐标尺寸 - 不要除以2！
            actual_maze_width = self.maze_grid.shape[1] * self.cell_size
            actual_maze_height = self.maze_grid.shape[0] * self.cell_size

            # 计算迷宫中心位置
            maze_center_x = actual_maze_width / 2
            maze_center_y = actual_maze_height / 2

            # 添加适当的边距确保完全覆盖
            margin = 2 * self.cell_size
            ground_width = actual_maze_width + 2 * margin
            ground_height = actual_maze_height + 2 * margin

            # 地面中心位置与迷宫中心对齐
            ground_center_x = maze_center_x
            ground_center_y = maze_center_y
        else:
            # 备用方案：使用原始尺寸
            maze_width = self.width * self.cell_size
            maze_height = self.height * self.cell_size
            margin = 2 * self.cell_size
            ground_width = maze_width + 2 * margin
            ground_height = maze_height + 2 * margin
            ground_center_x = maze_width / 2
            ground_center_y = maze_height / 2

        ground = self.scene.add_entity(
            gs.morphs.Plane(
                plane_size=(ground_width, ground_height),
                pos=(ground_center_x, ground_center_y, -0.1)  # 地面稍微低于迷宫，与迷宫中心对齐
            ),
            material=gs.materials.Rigid(
                rho=1000,  # 正常密度
                friction=0.1  # 降低摩擦力，让机器人能够移动
            ),
            surface=gs.surfaces.Default(
                color=(0.95, 0.95, 0.95)  # 非常浅的灰色，几乎透明
            )
        )

        # 添加迷宫墙壁
        self._add_walls_to_scene()

        # 不添加起点和终点标记

        return self.scene

    def _add_walls_to_scene(self):
        """将迷宫墙壁添加到Genesis场景中 - 使用连续墙体减少物体数量"""
        self.walls = []

        # 计算迷宫的整体尺寸
        maze_width = self.width * self.cell_size
        maze_height = self.height * self.cell_size
        wall_thickness = self.cell_size
        wall_height = self.cell_size

        # 移除边界墙体，只保留迷宫内部的复杂结构
        # 这样可以避免"矩形框地图"的效果，让迷宫看起来更自然

        # 对于内部墙体，我们将尝试合并连续的墙体来减少物体数量
        self._add_optimized_internal_walls()

    def _add_optimized_internal_walls(self):
        """添加优化的内部墙体 - 合并连续墙体减少物体数量"""

        # 首先找出所有需要墙体的位置
        wall_positions = []
        for y in range(self.maze_grid.shape[0]):
            for x in range(self.maze_grid.shape[1]):
                if self.maze_grid[y, x] == 1:  # 墙壁
                    wall_positions.append((x, y))

        # 处理过的位置，避免重复创建
        processed = set()

        for x, y in wall_positions:
            if (x, y) in processed:
                continue

            # 尝试创建水平连续墙体
            wall_length = 1
            while (x + wall_length < self.width and
                   (x + wall_length, y) in wall_positions and
                   (x + wall_length, y) not in processed):
                wall_length += 1

            if wall_length > 1:
                # 创建水平长墙
                wall_pos = (
                    (x + wall_length / 2 - 0.5) * self.cell_size,
                    y * self.cell_size,
                    self.cell_size / 2
                )
                wall_size = (wall_length * self.cell_size, self.cell_size, self.cell_size)

                wall = self.scene.add_entity(
                    gs.morphs.Box(
                        size=wall_size,
                        pos=wall_pos,
                        fixed=True  # 关键：设置为固定物体，不会移动
                    ),
                    material=gs.materials.Rigid(
                        rho=1000,  # 正常密度即可，因为fixed=True会覆盖物理属性
                        friction=0.8
                    ),
                    surface=gs.surfaces.Default(
                        color=(0.5, 0.5, 0.5)
                    )
                )
                self.walls.append(wall)

                # 标记这些位置为已处理
                for i in range(wall_length):
                    processed.add((x + i, y))
            else:
                # 尝试创建垂直连续墙体
                wall_height_cells = 1
                while (y + wall_height_cells < self.height and
                       (x, y + wall_height_cells) in wall_positions and
                       (x, y + wall_height_cells) not in processed):
                    wall_height_cells += 1

                # 创建单个或垂直墙体
                wall_pos = (
                    x * self.cell_size,
                    (y + wall_height_cells / 2 - 0.5) * self.cell_size,
                    self.cell_size / 2
                )
                wall_size = (self.cell_size, wall_height_cells * self.cell_size, self.cell_size)

                wall = self.scene.add_entity(
                    gs.morphs.Box(
                        size=wall_size,
                        pos=wall_pos,
                        fixed=True  # 关键：设置为固定物体，不会移动
                    ),
                    material=gs.materials.Rigid(
                        rho=1000,  # 正常密度即可，因为fixed=True会覆盖物理属性
                        friction=0.8
                    ),
                    surface=gs.surfaces.Default(
                        color=(0.5, 0.5, 0.5)
                    )
                )
                self.walls.append(wall)

                # 标记这些位置为已处理
                for i in range(wall_height_cells):
                    processed.add((x, y + i))

    def _add_start_goal_markers(self):
        """添加起点和终点标记"""
        if self.start_pos is None or self.goal_pos is None:
            return

        # 起点标记（绿色圆柱体，悬空，不碰撞）
        start_world_pos = (
            self.start_pos[0] * self.cell_size + self.cell_size / 2,
            self.start_pos[1] * self.cell_size + self.cell_size / 2,
            0.3  # 悬空高度，避免与小车碰撞
        )

        start_marker = self.scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.15,  # 较小半径
                height=0.4,  # 高度
                pos=start_world_pos,
                fixed=True  # 固定，不参与物理碰撞
            ),
            material=gs.materials.Rigid(
                rho=1,  # 极低密度，几乎无质量
                friction=0.01  # 最小摩擦值
            ),
            surface=gs.surfaces.Default(
                color=(0.0, 1.0, 0.0)  # 绿色
            )
        )

        # 终点标记（红色立方体，悬空，不碰撞）
        goal_world_pos = (
            self.goal_pos[0] * self.cell_size + self.cell_size / 2,
            self.goal_pos[1] * self.cell_size + self.cell_size / 2,
            0.3  # 悬空高度，避免与小车碰撞
        )

        goal_marker = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.3, 0.3, 0.4),  # 立方体尺寸
                pos=goal_world_pos,
                fixed=True  # 固定，不参与物理碰撞
            ),
            material=gs.materials.Rigid(
                rho=1,  # 极低密度，几乎无质量
                friction=0.01  # 最小摩擦值
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 0.0, 0.0)  # 红色
            )
        )

    def get_grid_position(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """
        将世界坐标转换为网格坐标

        Args:
            world_pos: 世界坐标 (x, y)

        Returns:
            网格坐标 (grid_x, grid_y)
        """
        grid_x = int(world_pos[0] / self.cell_size)
        grid_y = int(world_pos[1] / self.cell_size)
        return grid_x, grid_y

    def get_world_position(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """
        将网格坐标转换为世界坐标

        Args:
            grid_pos: 网格坐标 (grid_x, grid_y)

        Returns:
            世界坐标 (x, y)
        """
        world_x = grid_pos[0] * self.cell_size + self.cell_size / 2
        world_y = grid_pos[1] * self.cell_size + self.cell_size / 2
        return world_x, world_y

    def is_valid_position(self, grid_pos: Tuple[int, int]) -> bool:
        """
        检查网格位置是否有效（不是墙壁且在边界内）

        Args:
            grid_pos: 网格坐标 (grid_x, grid_y)

        Returns:
            位置是否有效
        """
        x, y = grid_pos
        if (x < 0 or x >= self.maze_grid.shape[1] or
                y < 0 or y >= self.maze_grid.shape[0]):
            return False
        return self.maze_grid[y, x] == 0

    def get_maze_bounds(self) -> Tuple[int, int]:
        """
        获取迷宫网格的实际边界

        Returns:
            (width, height) 迷宫网格的实际尺寸
        """
        if self.maze_grid is None:
            return self.width, self.height
        return self.maze_grid.shape[1], self.maze_grid.shape[0]

    def upscale_maze(self, scale: int = 2):
        """
        放大迷宫，使每个格子变成 scale x scale 的大格子

        Args:
            scale: 放大倍数，整数
        """
        if self.maze_grid is None:
            raise ValueError("请先生成迷宫网格")

        old_h, old_w = self.maze_grid.shape
        new_h, new_w = old_h * scale, old_w * scale

        new_maze = np.ones((new_h, new_w), dtype=int)  # 新迷宫初始化为全墙

        for y in range(old_h):
            for x in range(old_w):
                new_maze[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = self.maze_grid[y, x]

        self.maze_grid = new_maze

        # 更新起点和终点坐标
        self.start_pos = (self.start_pos[0] * scale, self.start_pos[1] * scale)
        self.goal_pos = (self.goal_pos[0] * scale, self.goal_pos[1] * scale)

        print(f"🔍 迷宫已放大 {scale} 倍，新尺寸: {new_w} x {new_h}")

    def get_neighbors(self, grid_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        获取有效邻居位置

        Args:
            grid_pos: 网格坐标 (grid_x, grid_y)

        Returns:
            有效邻居位置列表
        """
        x, y = grid_pos
        neighbors = []

        # 四个方向的邻居
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if self.is_valid_position(neighbor):
                neighbors.append(neighbor)

        return neighbors

    def build_scene(self):
        """构建Genesis场景"""
        if self.scene is None:
            raise ValueError("场景未创建，请先调用create_genesis_scene()")
        self.scene.build()

        # 使用fixed=True创建墙体，无需额外锁定
        print("🔒 墙体已使用fixed=True创建，完全静态！")

    def _lock_all_walls(self):
        """锁定所有墙体，确保它们完全静态"""
        import torch

        print("🔒 正在锁定所有墙体...")
        for i, wall in enumerate(self.walls):
            try:
                # 设置零速度
                if hasattr(wall, 'set_vel'):
                    wall.set_vel(torch.zeros(3, dtype=torch.float32))
                if hasattr(wall, 'set_ang'):
                    wall.set_ang(torch.zeros(3, dtype=torch.float32))

                # 如果支持，设置为动力学固定
                if hasattr(wall, 'set_kinematic'):
                    wall.set_kinematic(True)

                print(f"✅ 墙体 {i + 1}/{len(self.walls)} 已锁定")

            except Exception as e:
                print(f"⚠️  墙体 {i + 1} 锁定失败: {e}")

        print("🔒 所有墙体锁定完成！")

    def step(self):
        """执行一步仿真"""
        if self.scene is None:
            raise ValueError("场景未创建")
        self.scene.step()

    def get_scene(self) -> gs.Scene:
        """获取Genesis场景对象"""
        return self.scene
