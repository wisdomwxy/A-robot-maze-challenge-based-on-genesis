# A-robot-maze-challenge-based-on-genesis
Walking control and visualization demonstration of the Unitree Go2 quadruped robot based on the Genesis simulation platform, as well as the car searching for the finish line in the maze.
基于Genesis仿真平台的Unitree Go2四足机器人行走控制与可视化演示，以及小车在迷宫中寻找终点线。
# 中文版本
# 项目整体功能展示

![](./figure/2.gif?msec=1759325406919)

# 小车迷宫寻优（base）使用教程

**项目目录**

绿色框为自己新建文件夹

红色框为压缩包解压的

![](./figure/1.png?msec=1759325406919)


**终端**

![](./figure/2.png?msec=1759325406919)

激活环境，确保终端目录再上图的地址，在终端执行：

genesis-env\Scripts\activate

如果上述命令报错则执行一下命令

D:\pyc_workspace\RoboticsLessons-main\genesis-env\Scripts\Activate.ps1

如果出现红色报错，说无法执行脚本，那么以管理员身份打开powershell

执行：

set-executionpolicy remotesigned

根据提示直接输入**Y**即可

然后终端执行：

python maze_navigation_urdf_demo.py

如果报错**plane size**问题就需要升级genesis为最新版本

A*算法动态演示
![](./figure/1.gif?msec=1759325406919)
