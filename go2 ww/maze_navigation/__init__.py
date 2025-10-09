"""
机器人迷宫寻路项目
基于Genesis平台的机器人自主导航系统

主要功能：
- 自定义迷宫环境生成
- 机器人控制与导航
- 多种路径规划算法（A*、Dijkstra、强化学习）
- 实时可视化与调试
"""

__version__ = "1.0.0"
__author__ = "Genesis Robotics Team"

from .maze_environment import MazeEnvironment
from .robot_controller import RobotController
from .pathfinding import PathfindingAlgorithms

__all__ = [
    "MazeEnvironment",
    "RobotController", 
    "PathfindingAlgorithms"
]
