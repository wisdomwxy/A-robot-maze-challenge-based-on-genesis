# A-robot-maze-challenge-based-on-genesis
Walking control and visualization demonstration of the Unitree Go2 quadruped robot based on the Genesis simulation platform, as well as the car searching for the finish line in the maze.
基于Genesis仿真平台的Unitree Go2四足机器人行走控制与可视化演示，以及小车在迷宫中寻找终点线。
# 中文版本
# 小车迷宫寻优（base）使用教程

**项目目录**

绿色框为自己新建文件夹

红色框为压缩包解压的

![](./figure/1.png?msec=1759325406919)


**终端**

![](./figure/2.png?msec=1759325406919)

激活环境，确保终端目录再上图的地址，在终端执行：

genesis-env\Scripts\activate

如果出现红色报错，说无法执行脚本，那么以管理员身份打开powershell

执行：

set-executionpolicy remotesigned

根据提示直接输入**Y**即可

然后终端执行：

python maze_navigation_demo.py --maze-type recursive_backtracking --width 4 --height 4 --robot-type box

如果报错plane size问题就执行：

python maze_navigation_demo_genesis_0_2_1.py --maze-type recursive_backtracking --width 4 --height 4 --robot-type box

这个问题使因为genesis版本不一致引起的

保存为MP4运行视频
python maze_navigation_demo.py --maze-type recursive_backtracking --width 20 --height 20 --robot-type box --speed 1.0 --record-video --video-filename test_camera_angle.mp4

另一个版本：

python maze_navigation_demo_genesis_0_2_1.py --maze-type recursive_backtracking --width 20 --height 20 --robot-type box --speed 1.0 --record-video --video-filename test_camera_angle.mp4

如果录制的运行视频，视角太小，请调整**robot_controller.py**程序的170行左右的设置来改变相机位置

![](./figure/3.png?msec=1759325406919)

在videos路径下出现相应视频
