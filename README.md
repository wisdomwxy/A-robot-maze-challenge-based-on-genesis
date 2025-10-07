# A-robot-maze-challenge-based-on-genesis
Walking control and visualization demonstration of the Unitree Go2 quadruped robot based on the Genesis simulation platform, as well as the car searching for the finish line in the maze.  
基于Genesis仿真平台的Unitree Go2四足机器人行走控制与可视化演示，以及小车在迷宫中寻找终点线。  

## <1>Our research objectives:

**Objective 1:** Develop a highly flexible Genesis maze simulation environment.✔  
**Objective 2**: Implement autonomous path planning for robots, integrating and comparing five major algorithms (A*, Dijkstra, BFS, DFS, Q-Learning).✔  
**Objective 3**: Provide immersive 3D visualization and process recording.✔  
  
## <2>Future goals
1. Add the steering function of the car model
2. Train the go2 robot to find its way in a maze using reinforcement learning

## <3>Introduction to Path Optimization Algorithms
The A* (A-star) algorithm is one of the most popular and widely used pathfinding algorithms in computer science and artificial intelligence.   
It's renowned for its efficiency and ability to find the shortest path between a starting node and a goal node in a graph or grid.  

Dynamic demonstration of A* algorithm：  
![](./figure/1.gif?msec=1759325406919)
For the visualization of path optimization, please refer to the link:https://gallery.selfboot.cn/zh/algorithms  
## <4>We use the URDF model
**1.simple car**  

![](./figure/4.png?msec=1759325406919)

**2.go2 robot**

![](./figure/5.png?msec=1759325406919)

## <5>Visual display of program operation
**1.simple car**


![](./figure/2.gif?msec=1759325406919)

**2.go2 robot**


![](./figure/3.gif?msec=1759325406919)

We are currently unable to walk normally and are trying to solve the problem.

## <6>Quick project usage tutorial
![](./figure/2.png?msec=1759325406919)
Activate the environment and ensure that the terminal directory is at the address shown in the above picture. In the terminal, execute:  
**genesis-env\Scripts\activate**  
If the above command reports an error, execute the following command  
**D:\pyc_workspace\RoboticsLessons-main\genesis-env\Scripts\Activate.ps1**  
As you can see, our project is stored in the "pyc_workspace" path. Please adjust it according to your actual path.  
If a red error message appears indicating that the script cannot be executed, then open powershell as an administrator  
Execution：  
**set-executionpolicy remotesigned**  
Just enter **Y** directly as prompted  

Then the terminal executes:  
**python maze_navigation_urdf_demo.py**  
If the error "plane size" is reported, you need to upgrade genesis to the latest version (our genesis version is **0.3.3**).  

python maze_navigation_urdf_demo.py

如果报错**plane size**问题就需要升级genesis为最新版本

A*算法动态演示
![](./figure/1.gif?msec=1759325406919)
