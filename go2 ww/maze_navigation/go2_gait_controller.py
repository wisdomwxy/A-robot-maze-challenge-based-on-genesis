"""
Go2机器人步态控制器
集成真正的关节控制和四足机器人步态运动
基于Go2Env的关节控制逻辑
"""

import numpy as np
import torch
import genesis as gs
from typing import List, Tuple, Optional, Dict
import math
import os
import sys

# 优先使用包内导入，避免IDE无法解析
try:
    from locomotion.go2_env import Go2Env
except ImportError:
    print("⚠️ 警告: 无法导入Go2Env，将使用简化版本")
    Go2Env = None


class Go2GaitController:
    """Go2机器人步态控制器，实现真正的四足机器人行走"""

    def __init__(self, maze_env, record_video: bool = False, video_filename: str = None):
        """
        初始化Go2机器人步态控制器

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
        self.speed = 0.8  # 增加Go2机器人移动速度 (m/s) 以确保移动
        self.tolerance = 0.4  # 位置容差
        self.stuck_counter = 0
        self.max_stuck_steps = 300  # 进一步增加卡住判定步数，减少误判脱困
        
        # 摔倒恢复相关参数
        self.fall_recovery_steps = 0
        self.post_rotation_stabilize_steps = 0
        
        # Go2关节参数 - 基于Go2Env
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]
        
        # 改进的默认关节角度（更稳定的站立姿态）
        self.default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,    # 稍微增加大腿角度以提高稳定性
            "FR_thigh_joint": 0.9,    # 稍微增加大腿角度以提高稳定性
            "RL_thigh_joint": 1.0,    # 稍微增加大腿角度以提高稳定性
            "RR_thigh_joint": 1.0,    # 稍微增加大腿角度以提高稳定性
            "FL_calf_joint": -1.8,    # 调整小腿角度以提高稳定性
            "FR_calf_joint": -1.8,    # 调整小腿角度以提高稳定性
            "RL_calf_joint": -1.8,    # 调整小腿角度以提高稳定性
            "RR_calf_joint": -1.8,    # 调整小腿角度以提高稳定性
        }
        
        # 优化的PD控制器参数 - 更好的稳定性
        self.kp = 25.0  # 适中的刚度
        self.kd = 2.5   # 适中的阻尼
        
        # 步态参数 - 优化稳定性
        self.step_height = 0.05  # 减小步高，提高稳定性
        self.step_frequency = 2.5  # Hz - 降低步态频率以提高稳定性
        self.gait_phase = 0.0
        self.gait_cycle_time = 1.0 / self.step_frequency
        
        # 导航参数
        self.look_ahead_distance = 2.0  # 增大前瞻距离以改善路径跟踪
        self.smooth_factor = 0.8  # 增加平滑因子
        self.waypoint_threshold = 0.5  # 到达路径点的距离阈值
        self.is_navigation_finished = False
        
        # 转弯控制参数
        self.is_rotating = False
        self.rotation_target = 0.0
        self.rotation_speed = 0.1  # 降低转弯速度，提高稳定性
        self.rotation_threshold = 0.1  # 减小转弯完成的角度阈值（约5.7度）
        self.rotation_progress = 0.0  # 转弯进度
        
        # 动作延迟模拟 - 模拟真实机器人的1步延迟
        self.simulate_action_latency = True
        
        # 关节控制相关
        self.motors_dof_idx = None
        self.current_actions = torch.zeros(12, dtype=torch.float32)
        self.last_actions = torch.zeros(12, dtype=torch.float32)
        
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
        在Genesis场景中创建Go2机器人并设置关节控制

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
        # 使用Go2Env的标准初始高度
        robot_pos = (start_pos[0], start_pos[1], 0.42)  # Go2Env标准高度0.42m

        # 创建真正的Go2机器人 - 使用URDF模型
        try:
            # 计算初始朝向 - 面向第一个路径点
            initial_quat = self._calculate_initial_orientation(start_pos)
            
            self.robot = self.maze_env.scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/go2/urdf/go2.urdf",  # 真正的Go2机器人URDF文件
                    pos=robot_pos,
                    quat=initial_quat,  # 正确的初始朝向
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
            
            # 注意：关节控制设置在场景构建之后进行
            
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
            print("⚠️  使用简化模型，无法进行关节控制")

        self.current_pos = start_pos
        
        # 生成代价地图
        self._generate_cost_map()
        print(f"🗺️  Go2机器人代价地图已生成，大小: {self.cost_map_size}x{self.cost_map_size}")
        
        # 如果启用视频录制，设置视频录制
        if self.record_video:
            self._setup_video_recording()

        # 如果已有路径，调整机器人朝向
        if hasattr(self, 'path') and self.path and len(self.path) > 1:
            self._adjust_robot_orientation_after_creation()

        # 关键修复：在机器人创建后不立即设置关节控制，而是在场景构建完成后设置
        # 这确保了机器人实体已经完全构建
        if self.robot is not None and "URDF" in str(type(self.robot.morph)):
            print("⚙️  关节控制将在场景构建完成后设置")
        else:
            print("⚠️  无法设置关节控制：机器人不是URDF模型")

        return self.robot

    def _calculate_initial_orientation(self, start_pos: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """
        计算机器狗的初始朝向，使其面向第一个路径点
        
        Args:
            start_pos: 起始位置 (x, y)
            
        Returns:
            四元数 (x, y, z, w)
        """
        # 如果有路径，面向第一个目标点
        if hasattr(self, 'path') and self.path and len(self.path) > 1:
            first_target = self.maze_env.get_world_position(self.path[1])
            dx = first_target[0] - start_pos[0]
            dy = first_target[1] - start_pos[1]
            
            if abs(dx) > 0.01 or abs(dy) > 0.01:  # 避免除零
                target_angle = math.atan2(dy, dx)
                print(f"🎯 计算初始朝向: 面向目标点 {first_target}, 角度: {target_angle * 180 / math.pi:.1f}°")
            else:
                target_angle = 0.0
                print("🎯 使用默认朝向: 0°")
        else:
            # 默认朝向X轴正方向
            target_angle = 0.0
            print("🎯 使用默认朝向: 0°")
        
        # 将角度转换为四元数（绕Z轴旋转，航向角）
        quat_z = math.sin(target_angle / 2)
        quat_w = math.cos(target_angle / 2)
        return (0.0, 0.0, quat_z, quat_w)

    def _adjust_robot_orientation_after_creation(self):
        """机器人创建后调整朝向"""
        if self.robot is None or not hasattr(self, 'path') or not self.path or len(self.path) <= 1:
            return
            
        try:
            # 获取当前机器人位置
            current_pos = self.get_robot_position()
            
            # 计算到第一个目标点的方向
            first_target = self.maze_env.get_world_position(self.path[1])
            dx = first_target[0] - current_pos[0]
            dy = first_target[1] - current_pos[1]
            
            if abs(dx) > 0.01 or abs(dy) > 0.01:  # 避免除零
                print(f"🎯 机器人创建后调整朝向:")
                print(f"   当前位置: {current_pos}")
                print(f"   目标位置: {first_target}")
                print(f"   方向向量: ({dx:.3f}, {dy:.3f})")
                
                # 调整朝向
                self._adjust_robot_orientation(dx, dy)
            else:
                print("🎯 目标点太近，跳过朝向调整")
                
        except Exception as e:
            print(f"⚠️  机器人创建后朝向调整失败: {e}")

    def _setup_joint_control(self):
        """设置关节控制和PD控制器参数"""
        try:
            # 检查机器人是否已构建
            if hasattr(self.robot, 'is_built'):
                if callable(self.robot.is_built):
                    is_built = self.robot.is_built()
                else:
                    is_built = self.robot.is_built
                if not is_built:
                    print("⚠️  机器人实体尚未构建完成，无法设置关节控制")
                    return False
            else:
                # 如果没有is_built属性，尝试继续设置
                print("⚠️  无法确定机器人是否已构建，尝试继续设置关节控制")
                
            # 获取关节索引 - 使用正确的方法
            self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.joint_names]
            
            # 设置PD控制器参数 - 使用正确的API
            self.robot.set_dofs_kp([self.kp] * 12, self.motors_dof_idx)
            self.robot.set_dofs_kv([self.kd] * 12, self.motors_dof_idx)
            
            # 设置初始关节位置 - 使用正确的方法
            default_positions = torch.tensor([self.default_joint_angles[name] for name in self.joint_names], device=gs.device)
            self.robot.set_dofs_position(
                position=default_positions,
                dofs_idx_local=self.motors_dof_idx,
                zero_velocity=True
            )
            
            print("✅ 关节控制设置成功")
            return True
            
        except Exception as e:
            print(f"❌ 关节控制设置失败: {e}")
            import traceback
            traceback.print_exc()
            self.motors_dof_idx = None
            return False

    def _generate_gait_actions(self, direction: Tuple[float, float], speed: float) -> torch.Tensor:
        """
        生成有效的步态动作以驱动机器人移动
        
        Args:
            direction: 移动方向 (x, y)
            speed: 移动速度
            
        Returns:
            关节动作张量
        """
        # 更新步态相位
        self.gait_phase += self.step_frequency * 0.02  
        if self.gait_phase >= 2 * math.pi:
            self.gait_phase -= 2 * math.pi
        
        # 计算速度因子和移动意图
        speed_factor = min(speed / 0.5, 1.0)
        direction_magnitude = math.sqrt(direction[0]**2 + direction[1]**2)
        
        # 生成基础动作
        actions = torch.zeros(12, dtype=torch.float32)
        
        # 如果没有移动方向，返回站立姿态
        if direction_magnitude < 0.01:
            # 即使没有明确方向，也要保持步态以维持平衡
            direction = (1.0, 0.0)  # 默认向前
            direction_magnitude = 1.0
        
        # 计算目标方向角度
        target_angle = math.atan2(direction[1], direction[0])
        
        # 使用trot步态（对角腿同步移动）
        # 腿部顺序: FR(0), FL(1), RR(2), RL(3)
        leg_phases = [0, math.pi, math.pi, 0]  # 对角同步
        
        # 判断是否刚完成转身（在转身稳定期结束后的一段时间内）
        just_finished_rotation = self.post_rotation_stabilize_steps > -10 and self.post_rotation_stabilize_steps <= 0
        
        # 调整动作幅度以确保稳定性和移动效率的平衡
        if just_finished_rotation:
            # 刚完成转身，使用更小的动作幅度以确保平稳过渡
            base_amplitude = 0.5 * speed_factor
            forward_bias = 0.4 * speed_factor
        else:
            # 正常行走状态
            base_amplitude = 0.7 * speed_factor  # 增加基础幅度
            forward_bias = 0.5 * speed_factor    # 增加前进偏置
        
        # 根据移动方向调整转向
        turn_factor = max(-0.6, min(0.6, target_angle))  # 调整转向因子范围
        
        for leg_idx in range(4):
            leg_phase = self.gait_phase + leg_phases[leg_idx]
            base_idx = leg_idx * 3
            
            # Hip关节：控制转向和侧向移动
            if abs(turn_factor) > 0.1:  # 有明显转向需求
                if leg_idx == 0 or leg_idx == 2:  # 右侧腿
                    actions[base_idx] = turn_factor * 0.7  # 降低转向幅度以提高稳定性
                else:  # 左侧腿
                    actions[base_idx] = -turn_factor * 0.7
            else:  # 主要前进
                actions[base_idx] = 0.0
            
            # Thigh关节：主要推进力
            thigh_action = base_amplitude * math.sin(leg_phase) + forward_bias * math.cos(leg_phase)
            actions[base_idx + 1] = max(-1.3, min(1.3, thigh_action))
            
            # Calf关节：辅助推进和平衡
            calf_action = base_amplitude * 0.9 * math.sin(leg_phase + math.pi/4) + forward_bias * 0.7 * math.sin(leg_phase)
            actions[base_idx + 2] = max(-1.3, min(1.3, calf_action))
        
        return actions

    def _generate_rotation_actions(self, direction: int) -> torch.Tensor:
        """
        生成有效的原地转弯动作
        
        Args:
            direction: 转弯方向 (1为顺时针，-1为逆时针)
            
        Returns:
            关节动作张量
        """
        # 基础站立动作
        actions = torch.zeros(12, dtype=torch.float32)
        
        # 降低转弯幅度确保稳定转弯
        rotation_amplitude = 0.6 * direction  # 降低转弯幅度以提高稳定性
        
        # 使用对角步态进行转弯
        leg_phases = [0, math.pi, math.pi, 0]  # FR, FL, RR, RL
        
        for leg_idx in range(4):
            leg_phase = self.gait_phase + leg_phases[leg_idx]  # 使用当前步态相位
            base_idx = leg_idx * 3
            
            # Hip关节：转弯的主要驱动
            actions[base_idx] = rotation_amplitude * math.sin(leg_phase)
            
            # Thigh关节：保持稳定并辅助转弯
            actions[base_idx + 1] = 0.3 * math.cos(leg_phase)
            
            # Calf关节：保持平衡
            actions[base_idx + 2] = -0.3 * math.sin(leg_phase)
        
        return actions

    def _apply_joint_control(self, actions: torch.Tensor):
        """应用关节控制 - 使用Go2Env标准方式"""
        if self.motors_dof_idx is not None:
            try:
                # 确保动作张量在正确的设备上
                actions = actions.to(gs.device)
                
                # 存储当前动作
                self.current_actions = torch.clamp(actions, -1.5, 1.5)  # 适当放宽动作限制以确保移动
                
                # 选择执行的动作：模拟1步延迟（如果启用）
                exec_actions = self.last_actions if self.simulate_action_latency else self.current_actions
                
                # 计算目标关节位置 - 使用Go2Env标准方式
                default_positions = torch.tensor(
                    [self.default_joint_angles[name] for name in self.joint_names], 
                    device=gs.device
                )
                # 增加动作影响幅度以确保移动
                target_positions = exec_actions * 0.4 + default_positions  
                
                # 调试：打印关节控制信息
                if hasattr(self, '_joint_debug_counter'):
                    self._joint_debug_counter += 1
                else:
                    self._joint_debug_counter = 1
                    
                if self._joint_debug_counter % 100 == 0:  # 每100步打印一次
                    print(f"🦿 关节控制调试:")
                    print(f"   动作范围: [{actions.min():.3f}, {actions.max():.3f}]")
                    print(f"   目标位置范围: [{target_positions.min():.3f}, {target_positions.max():.3f}]")
                    print(f"   默认位置: {default_positions[:3]}")  # 只显示前3个关节
                
                # 应用关节控制 - 使用Go2Env标准方法
                self.robot.control_dofs_position(target_positions, self.motors_dof_idx)
                
                # 更新动作历史 - 在下一步开始时更新
                self.last_actions = self.current_actions.clone()
                
            except Exception as e:
                print(f"⚠️  关节控制失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️  关节索引未设置，无法应用关节控制")

    def _is_robot_fallen(self) -> bool:
        """
        检查机器人是否摔倒
        
        Returns:
            是否摔倒
        """
        try:
            # 获取机器人的姿态
            quat = self.robot.get_quat()
            if isinstance(quat, torch.Tensor):
                if quat.dim() > 1:
                    quat = quat[0]
                quat = quat.cpu().numpy()
            
            # 从四元数计算俯仰角和横滚角
            # 四元数格式为 [x, y, z, w]
            x, y, z, w = quat
            pitch = math.asin(2*(w*y - z*x))
            roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            
            # 如果俯仰角或横滚角超过45度，认为机器人摔倒了
            if abs(pitch) > math.pi/4 or abs(roll) > math.pi/4:
                print(f"🚨 机器人摔倒检测: 俯仰角={pitch*180/math.pi:.1f}°, 横滚角={roll*180/math.pi:.1f}°")
                return True
                
            return False
        except Exception as e:
            print(f"⚠️ 摔倒检测失败: {e}")
            return False

    def _recover_from_fall(self):
        """
        从摔倒状态恢复
        """
        try:
            print("🔄 执行摔倒恢复动作...")
            
            # 重置步态相位
            self.gait_phase = 0.0
            
            # 执行恢复动作序列
            for _ in range(50):  # 给机器人一些时间恢复
                # 使用默认站立姿态
                default_positions = torch.tensor(
                    [self.default_joint_angles[name] for name in self.joint_names], 
                    device=gs.device
                )
                
                # 应用站立姿态
                if self.motors_dof_idx is not None:
                    self.robot.control_dofs_position(default_positions, self.motors_dof_idx)
                
                # 执行仿真步骤
                self.maze_env.step()
            
            print("✅ 摔倒恢复动作完成")
            
            # 设置恢复期步数
            self.fall_recovery_steps = 30
            
        except Exception as e:
            print(f"❌ 摔倒恢复动作失败: {e}")

    def move_towards_target(self, dt: float = 0.02):
        """
        Go2机器人向目标位置移动，使用真正的步态控制
        """
        if self.robot is None or self.target_pos is None:
            return

        if self.is_path_complete():
            # 路径完成时保持站立姿态
            standing_actions = torch.zeros(12, dtype=torch.float32)
            self._apply_joint_control(standing_actions)
            print("🏁 路径已完成！")
            return
            
        # 如果正在转弯，执行转弯动作
        if self.is_rotating:
            self._execute_rotation(dt)
            return
            
        # 检查是否处于转身完成后的稳定期
        if self.post_rotation_stabilize_steps > 0:
            print(f"⏳ 转身稳定期剩余: {self.post_rotation_stabilize_steps} 步")
            # 在稳定期内保持站立姿态
            standing_actions = torch.zeros(12, dtype=torch.float32)
            self._apply_joint_control(standing_actions)
            self.post_rotation_stabilize_steps -= 1
            return

        # 检查是否处于摔倒恢复期
        if self.fall_recovery_steps > 0:
            print(f"🏥 摔倒恢复期剩余: {self.fall_recovery_steps} 步")
            # 在恢复期内保持站立姿态
            standing_actions = torch.zeros(12, dtype=torch.float32)
            self._apply_joint_control(standing_actions)
            self.fall_recovery_steps -= 1
            return

        current_pos = self.get_robot_position()
        
        # 检查机器人是否摔倒
        if self._is_robot_fallen():
            print("⚠️ 检测到机器人摔倒，开始恢复...")
            self._recover_from_fall()
            return

        # 获取前瞻目标点
        look_ahead_target = self._get_look_ahead_target()
        if look_ahead_target is not None:
            target = look_ahead_target
        elif self.target_pos is not None:
            target = self.target_pos
        else:
            # 如果没有前瞻目标点和当前目标点，使用路径中的下一个点
            if self.path_index < len(self.path):
                target = self.maze_env.get_world_position(self.path[self.path_index])
            else:
                # 路径已完成，保持站立姿态
                standing_actions = torch.zeros(12, dtype=torch.float32)
                self._apply_joint_control(standing_actions)
                print("🏁 路径已完成！")
                return

        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        
        # 添加调试信息
        if not hasattr(self, '_debug_move_counter'):
            self._debug_move_counter = 0
        self._debug_move_counter += 1
        
        if self._debug_move_counter % 100 == 0:
            print(f"📍 移动调试 {self._debug_move_counter}:")
            print(f"   当前路径索引: {self.path_index}/{len(self.path)-1}")
            print(f"   当前位置: ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
            print(f"   目标位置: ({target[0]:.2f}, {target[1]:.2f})")
            print(f"   距离目标: {distance:.2f}")
            print(f"   路径完成: {self.is_path_complete()}")
            if look_ahead_target:
                print(f"   前瞻目标: ({look_ahead_target[0]:.2f}, {look_ahead_target[1]:.2f})")

        if distance < self.tolerance:
            print(f"✅ 到达路径点 {self.path_index}: ({self.path[self.path_index]})")
            self.update_target()
            self.stuck_counter = 0
            # 到达目标点时使用站立姿态
            standing_actions = torch.zeros(12, dtype=torch.float32)
            self._apply_joint_control(standing_actions)
            return

        # 检查是否卡住 - 进一步放宽判定条件，减少误判
        if hasattr(self, '_last_pos'):
            # 增加位置变化阈值，减少因微小移动被判定为卡住的情况
            if abs(current_pos[0] - self._last_pos[0]) < 0.02 and abs(current_pos[1] - self._last_pos[1]) < 0.02:
                self.stuck_counter += 1
                # 添加调试信息
                if self.stuck_counter % 50 == 0:
                    pos_change = math.sqrt((current_pos[0] - self._last_pos[0])**2 + (current_pos[1] - self._last_pos[1])**2)
                    print(f"⚠️  位置变化检测: {pos_change:.4f}m (阈值: 0.02m), 卡住计数: {self.stuck_counter}")
            else:
                if self.stuck_counter > 0:
                    print(f"✅ 机器人正在移动，重置卡住计数器: {self.stuck_counter} -> 0")
                self.stuck_counter = 0
        else:
            self.stuck_counter = 0

        self._last_pos = current_pos

        # 如果卡住太久，尝试脱困
        if self.stuck_counter > self.max_stuck_steps:
            print(f"⚠️ 检测到机器人卡住 ({self.stuck_counter} 步)!")
            if self._try_escape_maneuver(current_pos):
                print("✅ 脱困机动成功")
                self.stuck_counter = 0
                return
            else:
                print("🔄 尝试更新目标点")
                self.update_target()
                self.stuck_counter = 0
                return

        # 计算移动方向
        direction_x, direction_y = self._get_safe_direction(current_pos, target)
        
        # 调试：打印方向信息
        if hasattr(self, 'step_count'):
            self.step_count += 1
        else:
            self.step_count = 1
            
        if self.step_count % 100 == 0:  # 每100步打印一次
            print(f"🔍 调试信息:")
            print(f"   当前位置: {current_pos}")
            print(f"   目标位置: {target}")
            print(f"   计算方向: ({direction_x:.3f}, {direction_y:.3f})")
            print(f"   距离: {distance:.3f}")
            print(f"   路径索引: {self.path_index}/{len(self.path)-1}")

        # 平滑转向
        if hasattr(self, '_last_direction'):
            direction_x = self.smooth_factor * direction_x + (1 - self.smooth_factor) * self._last_direction[0]
            direction_y = self.smooth_factor * direction_y + (1 - self.smooth_factor) * self._last_direction[1]
            norm = math.sqrt(direction_x ** 2 + direction_y ** 2)
            if norm > 0:
                direction_x /= norm
                direction_y /= norm

        self._last_direction = (direction_x, direction_y)

        # 生成步态动作 - 确保最小移动速度
        movement_speed = max(0.4, min(distance * 2, self.speed))  # 确保最小速度0.4，避免停止
        gait_actions = self._generate_gait_actions((direction_x, direction_y), movement_speed)
        
        # 应用关节控制
        self._apply_joint_control(gait_actions)

        # 注意：不进行刚体平移！
        # Go2机器人的移动应该完全通过关节控制和步态来实现
        # 机器人的位置会通过物理仿真自动更新

    def get_robot_position(self) -> Tuple[float, float]:
        """
        获取机器人当前位置（从物理仿真中获取真实位置）

        Returns:
            机器人位置 (x, y)
        """
        if self.robot is None:
            raise ValueError("机器人未创建")

        # 从物理仿真中获取真实的机器人位置
        pos = self.robot.get_pos()
        if isinstance(pos, torch.Tensor):
            if pos.dim() > 1:
                pos = pos[0]  # 取第一个环境的位置
            return (pos[0].item(), pos[1].item())
        else:
            return (pos[0], pos[1])

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
            # 调试：打印路径信息
            print(f"🤖 Go2机器人路径设置:")
            print(f"   路径长度: {len(path)}")
            print(f"   前5个路径点: {path[:5]}")
            print(f"   后5个路径点: {path[-5:]}")
            print(f"   起点: {path[0]}")
            print(f"   终点: {path[-1]}")
            
            # 设置第一个目标点（通常是起点）
            first_target_grid = path[0]
            self.target_pos = self.maze_env.get_world_position(first_target_grid)
            print(f"   当前目标点: {self.path_index} -> {first_target_grid}")
            print(f"   目标世界坐标: {self.target_pos}")
            
            # 如果机器人已创建，获取当前位置并计算方向
            if self.robot is not None:
                current_pos = self.get_robot_position()
                print(f"   当前机器人位置: {current_pos}")
                
                # 计算方向向量
                dx = self.target_pos[0] - current_pos[0]
                dy = self.target_pos[1] - current_pos[1]
                print(f"   方向向量: ({dx:.3f}, {dy:.3f})")
                print(f"   距离: {math.sqrt(dx*dx + dy*dy):.3f}m")
                
                # 动态调整机器狗朝向，使其与路径方向一致
                self._adjust_robot_orientation(dx, dy)
            else:
                print("   机器人未创建，将在创建后调整朝向")
        else:
            print("⚠️  警告: 路径为空!")
            self.target_pos = None

    def _adjust_robot_orientation(self, dx: float, dy: float):
        """
        根据路径方向动态调整机器狗朝向
        如果路径在后方，执行原地转弯
        
        Args:
            dx: X方向位移
            dy: Y方向位移
        """
        if self.robot is None:
            return
            
        # 获取机器人当前朝向
        current_quat = self.robot.get_quat()
        if isinstance(current_quat, torch.Tensor):
            if current_quat.dim() > 1:
                current_quat = current_quat[0]
            current_quat = current_quat.cpu().numpy()
        
        # 从四元数提取当前朝向角度（绕Z轴旋转，航向角）
        current_angle = 2 * math.atan2(current_quat[2], current_quat[3])
        
        # 计算目标方向角度（相对于X轴）
        target_angle = math.atan2(dy, dx)
        
        # 计算角度差
        angle_diff = target_angle - current_angle
        
        # 将角度差标准化到 [-π, π] 范围
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # 如果角度差大于30度，说明需要转向，执行原地转弯
        if abs(angle_diff) > math.pi / 6:  # 30度
            print(f"🔄 检测到需要转向，执行原地转弯:")
            print(f"   当前角度: {current_angle * 180 / math.pi:.1f}°")
            print(f"   目标角度: {target_angle * 180 / math.pi:.1f}°")
            print(f"   角度差: {angle_diff * 180 / math.pi:.1f}°")
            
            # 执行原地转弯 - 逐步旋转到目标角度
            self._perform_in_place_rotation(target_angle)
        else:
            # 小角度调整 - 允许机器狗在行走中逐步调整方向
            print(f"🚶 小角度调整，角度差: {angle_diff * 180 / math.pi:.1f}°")
            # 不进行原地转弯，让步态控制来处理小角度转向
            
    def _perform_in_place_rotation(self, target_angle: float):
        """
        执行原地转弯
        
        Args:
            target_angle: 目标角度
        """
        # 设置转弯模式
        self.is_rotating = True
        self.rotation_target = target_angle
        print("🔄 开始原地转弯...")
        
    def _set_robot_orientation(self, target_angle: float):
        """
        设置机器人朝向
        
        Args:
            target_angle: 目标角度
        """
        # 将角度转换为四元数（绕Z轴旋转，航向角）
        quat_z = math.sin(target_angle / 2)
        quat_w = math.cos(target_angle / 2)
        orientation_quat = (0.0, 0.0, quat_z, quat_w)
        
        # 应用新的朝向
        try:
            self.robot.set_quat(orientation_quat)
            print(f"🔄 机器狗朝向已调整:")
            print(f"   目标角度: {target_angle * 180 / math.pi:.1f}°")
            print(f"   四元数: {orientation_quat}")
        except Exception as e:
            print(f"⚠️  调整机器狗朝向失败: {e}")
            
    def _execute_rotation(self, dt: float):
        """
        执行稳定的原地转弯动作，使用真正的步态控制
        
        Args:
            dt: 时间步长
        """
        if not self.is_rotating:
            return
        
        # 更新步态相位
        self.gait_phase += self.step_frequency * dt
        if self.gait_phase >= 2 * math.pi:
            self.gait_phase -= 2 * math.pi
        
        # 更新转弯进度
        self.rotation_progress += dt
            
        # 获取机器人当前朝向
        current_quat = self.robot.get_quat()
        if isinstance(current_quat, torch.Tensor):
            if current_quat.dim() > 1:
                current_quat = current_quat[0]
            current_quat = current_quat.cpu().numpy()
        
        # 从四元数提取当前朝向角度（绕Z轴旋转，航向角）
        current_angle = 2 * math.atan2(current_quat[2], current_quat[3])
        
        # 计算角度差
        angle_diff = self.rotation_target - current_angle
        
        # 将角度差标准化到 [-π, π] 范围
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # 检查是否完成转弯
        if abs(angle_diff) < self.rotation_threshold:
            print("✅ 原地转弯完成！")
            self.is_rotating = False
            self.rotation_target = 0.0
            self.rotation_progress = 0.0
            
            # 设置转身完成后的稳定期步数
            self.post_rotation_stabilize_steps = 20
            
            # 停止时使用站立姿态
            standing_actions = torch.zeros(12, dtype=torch.float32)
            self._apply_joint_control(standing_actions)
            return
        
        # 计算转弯方向和剩余距离
        rotation_direction = 1 if angle_diff > 0 else -1
        
        # 根据剩余角度调整转弯速度 - 实现渐进减速
        remaining_angle_ratio = min(abs(angle_diff) / (math.pi/2), 1.0)  # 归一化到 [0,1]
        speed_factor = max(0.5, remaining_angle_ratio)  # 最小速度为0.5，避免停止
        
        # 生成转弯动作
        rotation_actions = self._generate_rotation_actions(rotation_direction)
        
        # 应用关节控制
        self._apply_joint_control(rotation_actions)
        
        # 调试信息
        if hasattr(self, '_rotation_debug_counter'):
            self._rotation_debug_counter += 1
        else:
            self._rotation_debug_counter = 1
            
        if self._rotation_debug_counter % 50 == 0:  # 每50步打印一次
            print(f"🔄 转弯进度:")
            print(f"   当前角度: {current_angle * 180 / math.pi:.1f}°")
            print(f"   目标角度: {self.rotation_target * 180 / math.pi:.1f}°")
            print(f"   角度差: {angle_diff * 180 / math.pi:.1f}°")
            print(f"   转弯方向: {'顺时针' if rotation_direction > 0 else '逆时针'}")
            print(f"   速度因子: {speed_factor:.2f}")
            print(f"   进度: {self.rotation_progress:.2f}s")

    def update_target(self):
        """更新目标位置到路径中的下一个点"""
        print(f"🔄 更新目标点: 当前索引 {self.path_index}")
        
        if self.path_index < len(self.path) - 1:
            self.path_index += 1
            self.target_pos = self.maze_env.get_world_position(self.path[self.path_index])
            
            print(f"   新目标点: 索引 {self.path_index} -> {self.path[self.path_index]}")
            print(f"   世界坐标: {self.target_pos}")
            
            # 更新朝向以面向新目标
            current_pos = self.get_robot_position()
            dx = self.target_pos[0] - current_pos[0]
            dy = self.target_pos[1] - current_pos[1]
            self._adjust_robot_orientation(dx, dy)
        else:
            # 已经是最后一个路径点
            self.target_pos = None
            if len(self.path) > 0:
                # 确保最终目标点是路径的最后一个点
                self.target_pos = self.maze_env.get_world_position(self.path[-1])
            print("   已到达路径终点")

    def is_at_target(self) -> bool:
        """
        检查机器人是否到达目标位置

        Returns:
            是否到达目标
        """
        if self.target_pos is None:
            print("⚠️  目标位置为None")
            return True

        current_pos = self.get_robot_position()
        distance = math.sqrt(
            (current_pos[0] - self.target_pos[0]) ** 2 +
            (current_pos[1] - self.target_pos[1]) ** 2
        )
        
        # 添加调试信息
        if not hasattr(self, '_debug_target_counter'):
            self._debug_target_counter = 0
        self._debug_target_counter += 1
        
        if self._debug_target_counter % 50 == 0:
            print(f"🎯 目标检测: 当前位置({current_pos[0]:.2f}, {current_pos[1]:.2f}), "
                  f"目标位置({self.target_pos[0]:.2f}, {self.target_pos[1]:.2f}), "
                  f"距离={distance:.2f}, 容差={self.tolerance}, 到达={distance < self.tolerance}")
        
        is_at = distance < self.tolerance
        return is_at

    def is_path_complete(self) -> bool:
        """
        检查路径是否完成

        Returns:
            路径是否完成
        """
        # 修复路径完成判断逻辑
        if len(self.path) == 0:
            return True
            
        # 检查是否已经访问完所有路径点
        if self.path_index >= len(self.path):
            return True
            
        # 检查是否到达最后一个路径点
        if self.path_index == len(self.path) - 1:
            current_pos = self.get_robot_position()
            last_waypoint = self.maze_env.get_world_position(self.path[-1])
            distance_to_last = math.sqrt(
                (current_pos[0] - last_waypoint[0])**2 + 
                (current_pos[1] - last_waypoint[1])**2
            )
            return distance_to_last < self.tolerance
            
        return False

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
        """尝试脱困机动 - 通过步态控制而不是刚体平移"""
        escape_directions = [
            (1.0, 0), (0, 1.0), (-1.0, 0), (0, -1.0),
            (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)
        ]

        for i, (dx, dy) in enumerate(escape_directions):
            # 计算脱困目标位置
            escape_target = (
                current_pos[0] + dx * 0.5,  # 减小脱困距离
                current_pos[1] + dy * 0.5
            )

            # 检查脱困位置是否安全
            if self._is_position_safe(escape_target):
                print(f"🚀 Go2机器人脱困机动: 方向 {i + 1}")
                # 生成脱困步态动作
                escape_actions = self._generate_gait_actions((dx, dy), self.speed * 1.5)  # 脱困时速度更快
                self._apply_joint_control(escape_actions)
                
                # 执行几步以确保动作生效
                for _ in range(10):  # 增加执行步数
                    self.maze_env.step()
                    
                return True

        return False

    def _get_look_ahead_target(self) -> Optional[Tuple[float, float]]:
        """获取前瞻目标点"""
        if self.path_index >= len(self.path) - 1:
            return None

        current_pos = self.get_robot_position()
        if not current_pos:
            return None

        # 如果当前路径点索引已经超出路径长度，返回最后一个点
        if self.path_index >= len(self.path):
            return self.maze_env.get_world_position(self.path[-1])

        for i in range(self.path_index, len(self.path)):  # 从当前路径点开始查找
            target_world = self.maze_env.get_world_position(self.path[i])
            distance = math.sqrt(
                (target_world[0] - current_pos[0]) ** 2 +
                (target_world[1] - current_pos[1]) ** 2
            )

            # 如果距离大于前瞻距离，返回前一个点作为目标
            if distance > self.look_ahead_distance:
                # 如果i是第一个点，返回它；否则返回前一个点
                if i == self.path_index:
                    return target_world
                else:
                    return self.maze_env.get_world_position(self.path[i - 1])
        
        # 如果没有找到距离大于前瞻距离的点，返回路径的最后一个点
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


    def get_robot_entity(self):
        """获取机器人实体对象"""
        return self.robot

    def _setup_video_recording(self):
        """设置视频录制"""
        if self.video_filename is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_filename = f"go2_gait_maze_navigation_{timestamp}.mp4"

        # 确保视频文件保存在videos文件夹中
        if not self.video_filename.startswith("videos/"):
            self.video_filename = f"videos/{self.video_filename}"

        print(f"🎥 Go2机器人步态视频录制已启用: {self.video_filename}")

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
