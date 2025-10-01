#!/usr/bin/env python3
"""
Go2机器人步态迷宫导航演示程序
基于Genesis平台的Go2四足机器人真正步态行走系统

使用方法：
1. 激活Genesis环境：genesis-env\Scripts\activate
2. 运行演示：python maze_navigation_go2_gait_demo.py

功能特性：
- 真正的Go2机器人URDF模型
- 真实的四足机器人步态控制
- 关节控制和PD控制器
- 多种迷宫生成算法
- 多种路径规划算法
- 实时3D可视化
- 步态行走动画
"""

import argparse
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maze_navigation import MazeEnvironment, PathfindingAlgorithms
from maze_navigation.go2_gait_controller import Go2GaitController


def print_banner():
    """打印程序横幅"""
    print("=" * 60)
    print("🐕 Go2机器人步态迷宫寻路系统 - Genesis平台")
    print("=" * 60)
    print("功能特性：")
    print("• 真正的Go2四足机器人URDF模型")
    print("• 真实的关节控制和步态行走")
    print("• PD控制器和步态算法")
    print("• 自定义迷宫环境生成")
    print("• 多种路径规划算法")
    print("• 实时3D可视化")
    print("• 四足机器人行走动画")
    print("=" * 60)


def create_go2_gait_maze_demo(maze_type: str = "recursive_backtracking",
                             width: int = 10, height: int = 8,
                             show_viewer: bool = True, build_scene: bool = True,
                             record_video: bool = False, video_filename: str = None):
    """
    创建Go2机器人步态迷宫演示

    Args:
        maze_type: 迷宫类型
        width: 迷宫宽度
        height: 迷宫高度
        show_viewer: 是否显示可视化
        build_scene: 是否立即构建场景
        record_video: 是否录制视频
        video_filename: 视频文件名（可选）
    """
    print(f"\n🏗️  创建 {width}x{height} 步态迷宫 ({maze_type})...")

    # 创建迷宫环境 - 为Go2机器人调整尺寸
    maze_env = MazeEnvironment(width=width, height=height, cell_size=1.5)  # 增大格子尺寸

    # 生成迷宫
    maze_grid = maze_env.generate_maze(algorithm=maze_type)
    print(f"✅ 迷宫生成完成")
    print(f"   起点: {maze_env.start_pos}")
    print(f"   终点: {maze_env.goal_pos}")
    
    # 放大迷宫（每格放大 scale x scale）
    maze_env.upscale_maze(scale=2)

    # 创建Genesis场景
    scene = maze_env.create_genesis_scene(show_viewer=show_viewer,
                                        record_video=record_video,
                                        video_filename=video_filename)

    if build_scene:
        maze_env.build_scene()

    return maze_env, scene


def demo_go2_gait_navigation(maze_env, path, show_viewer: bool = True,
                            record_video: bool = False, video_filename: str = None):
    """
    演示Go2机器人步态导航

    Args:
        maze_env: 迷宫环境对象
        path: 导航路径
        show_viewer: 是否显示可视化
        record_video: 是否录制视频
        video_filename: 视频文件名（可选）
    """
    if not path:
        print("❌ 没有有效路径，跳过Go2机器人步态导航演示")
        return

    print(f"\n🐕 Go2机器人步态导航演示...")
    print(f"   路径长度: {len(path)} 步")

    # 创建Go2机器人步态控制器
    robot_controller = Go2GaitController(maze_env, record_video=record_video, video_filename=video_filename)

    # 创建Go2机器人（在场景构建之前）
    robot = robot_controller.create_robot()
    print("✅ Go2机器人步态控制器创建完成")

    # 构建场景（机器人已添加）
    maze_env.build_scene()
    print("✅ 场景构建完成")
    
    # 场景构建后设置关节控制
    robot_controller._setup_joint_control()

    # 设置路径
    robot_controller.set_path(path)
    print("✅ 路径设置完成")

    # 开始导航
    print("🚀 开始步态导航...")
    print("   按 Ctrl+C 停止演示")

    step_count = 0
    start_time = time.time()

    # 开始视频录制
    if record_video and hasattr(robot_controller, 'camera'):
        robot_controller.camera.start_recording()
        print("🎥 步态视频录制已开始")

    try:
        while not robot_controller.is_path_complete():
            # 移动Go2机器人（使用真正的步态控制）
            robot_controller.move_towards_target(dt=0.02)  # 50Hz控制频率，匹配Go2Env

            # 执行仿真步骤
            maze_env.step()

            # 如果录制视频，渲染相机图像
            if record_video and hasattr(robot_controller, 'camera') and hasattr(robot_controller, '_render_camera'):
                robot_controller.camera.render()

            step_count += 1

            # 每100步显示进度
            if step_count % 100 == 0:
                current_pos = robot_controller.get_robot_grid_position()
                distance_to_goal = robot_controller.get_distance_to_goal()
                print(f"   步骤 {step_count}: 当前位置 {current_pos}, 距离终点 {distance_to_goal:.2f}m")

            # 检查是否到达终点
            if robot_controller.is_at_goal():
                break

        end_time = time.time()
        print(f"\n🎉 步态导航完成!")
        print(f"   总步数: {step_count}")
        print(f"   总时间: {end_time - start_time:.2f} 秒")
        print(f"   平均速度: {step_count * 0.01 / (end_time - start_time):.2f} 步/秒")

        # 停止视频录制
        if record_video and hasattr(robot_controller, 'camera'):
            robot_controller.camera.stop_recording(save_to_filename=robot_controller.video_filename, fps=30)
            print(f"🎥 步态视频已保存: {robot_controller.video_filename}")

    except KeyboardInterrupt:
        print(f"\n⏹️  演示被用户中断")
        print(f"   已完成步数: {step_count}")

        # 停止视频录制
        if record_video and hasattr(robot_controller, 'camera'):
            robot_controller.camera.stop_recording(save_to_filename=robot_controller.video_filename, fps=30)
            print(f"🎥 步态视频已保存: {robot_controller.video_filename}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Go2机器人步态迷宫寻路演示程序")
    parser.add_argument("--maze-type", default="recursive_backtracking",
                       choices=["recursive_backtracking", "random", "simple", "open"],
                       help="迷宫生成算法")
    parser.add_argument("--width", type=int, default=8, help="迷宫宽度")
    parser.add_argument("--height", type=int, default=6, help="迷宫高度")
    parser.add_argument("--no-viewer", action="store_true", help="不显示可视化窗口")
    parser.add_argument("--record-video", action="store_true", help="录制视频并保存到videos文件夹")
    parser.add_argument("--video-filename", type=str, default=None, help="视频文件名（可选，默认自动生成）")

    args = parser.parse_args()

    print_banner()

    try:
        # 创建迷宫（不立即构建场景）
        maze_env, scene = create_go2_gait_maze_demo(
            maze_type=args.maze_type,
            width=args.width,
            height=args.height,
            show_viewer=not args.no_viewer,
            build_scene=False,
            record_video=args.record_video,
            video_filename=args.video_filename
        )

        # 路径规划
        print(f"\n🧭 路径规划算法演示...")
        # 为Go2机器人调整路径规划参数
        pathfinder = PathfindingAlgorithms(maze_env, robot_width=0.35, robot_height=0.35, safety_margin=0.15)
        start = maze_env.start_pos
        goal = maze_env.goal_pos

        if start is None or goal is None:
            print("❌ 起点或终点未设置")
            return

        print(f"   起点: {start}")
        print(f"   终点: {goal}")

        # 定义四个算法
        algorithms = {
            "A*": pathfinder.a_star,
            "Dijkstra": pathfinder.dijkstra,
            "BFS": pathfinder.breadth_first_search,
            "DFS": pathfinder.depth_first_search
        }
        results = {}

        # 循环执行每个算法并打印结果
        for name, func in algorithms.items():
            print(f"\n🔍 运行 {name} 算法...")
            start_time = time.time()
            path = func(start, goal)
            end_time = time.time()

            if path:
                path_length = pathfinder.get_path_length(path)
                print(f"   ✅ 找到路径，长度: {len(path)} 步，距离: {path_length:.2f}")
                print(f"   ⏱️ 计算时间: {(end_time - start_time) * 1000:.2f} ms")
                smoothed_path = pathfinder.smooth_path(path)
                results[name] = smoothed_path
            else:
                print(f"   ❌ 未找到路径")
                results[name] = None

        # 选择一个算法路径进行步态导航
        selected_algorithm = "A*"  # 可改为 "A*","Dijkstra", "BFS", "DFS"
        path_to_follow = results[selected_algorithm]

        if path_to_follow:
            demo_go2_gait_navigation(
                maze_env, path_to_follow,
                show_viewer=not args.no_viewer,
                record_video=args.record_video,
                video_filename=args.video_filename
            )
        else:
            print(f"   ❌ 未找到路径")

        print(f"\n🎉 步态演示完成!")

    except KeyboardInterrupt:
        print(f"\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print(f"请确保:")
        print(f"1. Genesis环境已激活: genesis-env\\Scripts\\activate")
        print(f"2. 所有依赖已安装: pip install -r requirements.txt")
        print(f"3. Go2 URDF文件存在: urdf/go2/urdf/go2.urdf")
        sys.exit(1)


if __name__ == "__main__":
    main()
