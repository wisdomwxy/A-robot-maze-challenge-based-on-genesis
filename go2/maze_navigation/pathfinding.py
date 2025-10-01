"""
路径规划算法类
实现A*、Dijkstra等路径规划算法
"""

import numpy as np
import heapq
import random
from typing import List, Tuple, Optional, Dict, Set
import math
from collections import deque


class PathfindingAlgorithms:
    """路径规划算法集合类"""

    def __init__(self, maze_env, robot_width=0.36, robot_height=0.36, safety_margin=0.1):
        """
        初始化路径规划算法

        Args:
            maze_env: 迷宫环境对象
            robot_width: 机器人宽度
            robot_height: 机器人高度
            safety_margin: 安全边距
        """
        self.maze_env = maze_env
        self.robot_width = robot_width
        self.robot_height = robot_height
        self.safety_margin = safety_margin
        self.effective_width = robot_width + 2 * safety_margin
        self.effective_height = robot_height + 2 * safety_margin

    def _is_position_safe_for_robot(self, grid_pos: Tuple[int, int]) -> bool:
        """
        检查网格位置是否对机器人安全，考虑机器人尺寸

        Args:
            grid_pos: 网格坐标 (grid_x, grid_y)

        Returns:
            位置是否对机器人安全
        """
        x, y = grid_pos

        # 检查边界
        if (x < 0 or x >= self.maze_env.maze_grid.shape[1] or
                y < 0 or y >= self.maze_env.maze_grid.shape[0]):
            return False

        # 计算机器人占用的网格范围
        # 将机器人尺寸转换为网格单位
        cell_size = self.maze_env.cell_size
        robot_width_cells = math.ceil(self.effective_width / cell_size)
        robot_height_cells = math.ceil(self.effective_height / cell_size)

        # 检查机器人占用的所有网格
        half_width_cells = robot_width_cells // 2
        half_height_cells = robot_height_cells // 2

        for dx in range(-half_width_cells, half_width_cells + 1):
            for dy in range(-half_height_cells, half_height_cells + 1):
                check_x = x + dx
                check_y = y + dy

                # 检查边界
                if (check_x < 0 or check_x >= self.maze_env.maze_grid.shape[1] or
                        check_y < 0 or check_y >= self.maze_env.maze_grid.shape[0]):
                    return False

                # 检查是否为墙壁
                if self.maze_env.maze_grid[check_y, check_x] == 1:
                    return False

        return True

    def _get_safe_neighbors(self, grid_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        获取对机器人安全的邻居位置

        Args:
            grid_pos: 网格坐标 (grid_x, grid_y)

        Returns:
            安全邻居位置列表
        """
        x, y = grid_pos
        neighbors = []

        # 四个方向的邻居
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if self._is_position_safe_for_robot(neighbor):
                neighbors.append(neighbor)

        return neighbors

    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        A*算法路径规划

        Args:
            start: 起点网格坐标 (grid_x, grid_y)
            goal: 终点网格坐标 (grid_x, grid_y)

        Returns:
            路径点列表，如果无解则返回空列表
        """
        # 获取迷宫实际边界
        maze_width, maze_height = self.maze_env.get_maze_bounds()

        # 检查起点和终点是否在迷宫边界内
        if (start[0] < 0 or start[0] >= maze_width or start[1] < 0 or start[1] >= maze_height or
                goal[0] < 0 or goal[0] >= maze_width or goal[1] < 0 or goal[1] >= maze_height):
            print(f"⚠️  起点或终点超出迷宫边界: 起点{start}, 终点{goal}, 迷宫尺寸({maze_width}, {maze_height})")
            return []

        if not self._is_position_safe_for_robot(start) or not self._is_position_safe_for_robot(goal):
            print(f"⚠️  起点或终点位置对机器人不安全: 起点{start}, 终点{goal}")
            return []

        # 优先队列：(f_score, g_score, position, path)
        open_set = [(0, 0, start, [start])]
        closed_set = set()
        g_scores = {start: 0}

        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)

            if current == goal:
                return path

            # 检查所有邻居，使用机器人尺寸感知的检查
            for neighbor in self._get_safe_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_score + 1

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h_score = self._heuristic(neighbor, goal)
                    f_score = tentative_g + h_score

                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, new_path))

        return []  # 无解

    def dijkstra(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Dijkstra算法路径规划

        Args:
            start: 起点网格坐标 (grid_x, grid_y)
            goal: 终点网格坐标 (grid_x, grid_y)

        Returns:
            路径点列表，如果无解则返回空列表
        """
        # 获取迷宫实际边界
        maze_width, maze_height = self.maze_env.get_maze_bounds()

        # 检查起点和终点是否在迷宫边界内
        if (start[0] < 0 or start[0] >= maze_width or start[1] < 0 or start[1] >= maze_height or
                goal[0] < 0 or goal[0] >= maze_width or goal[1] < 0 or goal[1] >= maze_height):
            return []

        if not self._is_position_safe_for_robot(start) or not self._is_position_safe_for_robot(goal):
            return []

        # 优先队列：(distance, position, path)
        open_set = [(0, start, [start])]
        closed_set = set()
        distances = {start: 0}

        while open_set:
            distance, current, path = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)

            if current == goal:
                return path

            # 检查所有邻居，使用机器人尺寸感知的检查
            for neighbor in self._get_safe_neighbors(current):
                if neighbor in closed_set:
                    continue

                new_distance = distance + 1

                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (new_distance, neighbor, new_path))

        return []  # 无解

    def breadth_first_search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        广度优先搜索算法

        Args:
            start: 起点网格坐标 (grid_x, grid_y)
            goal: 终点网格坐标 (grid_x, grid_y)

        Returns:
            路径点列表，如果无解则返回空列表
        """
        # 获取迷宫实际边界
        maze_width, maze_height = self.maze_env.get_maze_bounds()

        # 检查起点和终点是否在迷宫边界内
        if (start[0] < 0 or start[0] >= maze_width or start[1] < 0 or start[1] >= maze_height or
                goal[0] < 0 or goal[0] >= maze_width or goal[1] < 0 or goal[1] >= maze_height):
            return []

        if not self._is_position_safe_for_robot(start) or not self._is_position_safe_for_robot(goal):
            return []

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            if current == goal:
                return path

            for neighbor in self._get_safe_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []  # 无解

    def depth_first_search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        深度优先搜索算法

        Args:
            start: 起点网格坐标 (grid_x, grid_y)
            goal: 终点网格坐标 (grid_x, grid_y)

        Returns:
            路径点列表，如果无解则返回空列表
        """
        # 获取迷宫实际边界
        maze_width, maze_height = self.maze_env.get_maze_bounds()

        # 检查起点和终点是否在迷宫边界内
        if (start[0] < 0 or start[0] >= maze_width or start[1] < 0 or start[1] >= maze_height or
                goal[0] < 0 or goal[0] >= maze_width or goal[1] < 0 or goal[1] >= maze_height):
            return []

        if not self._is_position_safe_for_robot(start) or not self._is_position_safe_for_robot(goal):
            return []

        stack = [(start, [start])]
        visited = {start}

        while stack:
            current, path = stack.pop()

            if current == goal:
                return path

            for neighbor in self._get_safe_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))

        return []  # 无解

    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        启发式函数（曼哈顿距离）

        Args:
            pos1: 位置1 (grid_x, grid_y)
            pos2: 位置2 (grid_x, grid_y)

        Returns:
            启发式距离
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_all_paths(self, start: Tuple[int, int], goal: Tuple[int, int],
                      max_paths: int = 10) -> List[List[Tuple[int, int]]]:
        """
        获取从起点到终点的多条路径

        Args:
            start: 起点网格坐标
            goal: 终点网格坐标
            max_paths: 最大路径数量

        Returns:
            路径列表
        """
        paths = []

        # 使用不同的算法获取路径
        algorithms = [
            ("A*", self.a_star),
            ("Dijkstra", self.dijkstra),
            ("BFS", self.breadth_first_search),
            ("DFS", self.depth_first_search)
        ]

        for name, algorithm in algorithms:
            path = algorithm(start, goal)
            if path and path not in paths:
                paths.append(path)
                if len(paths) >= max_paths:
                    break

        return paths

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        路径平滑化，非常保守地移除不必要的中间点

        Args:
            path: 原始路径

        Returns:
            平滑后的路径
        """
        if len(path) <= 2:
            return path

        # 在复杂迷宫中，使用非常保守的平滑策略
        # 只移除明显共线的点，避免过度简化
        smoothed = [path[0]]
        i = 0

        while i < len(path) - 1:
            # 只尝试跳过1个点，在复杂迷宫中保持更多路径点
            max_skip = min(2, len(path) - i - 1)  # 最多跳过2个点
            best_j = i + 1

            for j in range(i + 2, min(i + max_skip + 1, len(path))):
                # 检查从当前点到j点是否有直线路径且安全
                if self._is_direct_path_safe(smoothed[-1], path[j]):
                    best_j = j
                else:
                    # 如果直线路径不安全，立即停止尝试更远的点
                    break

            # 添加最佳可达点
            if best_j > i + 1:
                smoothed.append(path[best_j])
                i = best_j
            else:
                smoothed.append(path[i + 1])
                i += 1

        # 确保终点被包含
        if smoothed[-1] != path[-1]:
            smoothed.append(path[-1])

        return smoothed

    def _is_direct_path_safe(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """
        检查两点间的直线路径是否安全（考虑机器人尺寸）

        Args:
            start: 起点
            end: 终点

        Returns:
            路径是否安全
        """
        # 使用Bresenham算法检查直线路径上的所有点
        x0, y0 = start
        x1, y1 = end

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            # 检查当前点是否对机器人安全
            if not self._is_position_safe_for_robot((x, y)):
                return False

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return True

    def _is_collinear(self, p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> bool:
        """
        检查三点是否共线

        Args:
            p1, p2, p3: 三个点

        Returns:
            是否共线
        """
        # 计算向量
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # 检查叉积是否为0（共线）
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        return abs(cross_product) < 1e-6

    def get_path_length(self, path: List[Tuple[int, int]]) -> float:
        """
        计算路径长度

        Args:
            path: 路径点列表

        Returns:
            路径长度
        """
        if len(path) <= 1:
            return 0.0

        total_length = 0.0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            total_length += math.sqrt(dx ** 2 + dy ** 2)

        return total_length

    def compare_algorithms(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Dict:
        """
        比较不同算法的性能

        Args:
            start: 起点网格坐标
            goal: 终点网格坐标

        Returns:
            算法比较结果字典
        """
        import time

        algorithms = {
            "A*": self.a_star,
            "Dijkstra": self.dijkstra,
            "BFS": self.breadth_first_search,
            "DFS": self.depth_first_search
        }

        results = {}

        for name, algorithm in algorithms.items():
            start_time = time.time()
            path = algorithm(start, goal)
            end_time = time.time()

            results[name] = {
                "path": path,
                "length": len(path) if path else 0,
                "distance": self.get_path_length(path) if path else float('inf'),
                "time": end_time - start_time,
                "success": len(path) > 0
            }

        return results