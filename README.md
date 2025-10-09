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
Install relevant dependencies：

```bash
python3 setup.py
```
or
```bash
./setup.py
```

The script will automatically execute all the above steps and output the result of each step in the command line (✅ indicates success, ❌ indicates failure, ⚠️ indicates warning). If you encounter a ❌ Error, you need to correct it based on the error message (for example, Python version too low).


![](./figure/2.png?msec=1759325406919)  

Activate the environment and ensure that the terminal directory is at the address shown in the above picture. In the terminal, execute:  

```bash
genesis-env\Scripts\activate
```

If the above command reports an error, execute the following command  
```bash
D:\pyc_workspace\RoboticsLessons-main\genesis-env\Scripts\Activate.ps1
```

As you can see, our project is stored in the "pyc_workspace" path. Please adjust it according to your actual path.  
If a red error message appears indicating that the script cannot be executed, then open powershell as an administrator  
Execution： 

```bash
set-executionpolicy remotesigned
```
Just enter **Y** directly as prompted  

Because there are two small projects in our project.  
So to execute the corresponding code, you need to enter the folder where the small project is located, for example:

```bash
cd simple_car
```

or

```bash
cd robot_go2_github_version
```

Then the terminal executes(simple_car): 

```bash
python maze_navigation_urdf_demo.py
```

Or you can type the following code in the terminal(robot_go2_github_version)：

```bash
maze_navigation_go2_gait_demo.py
```
If the error "plane size" is reported, you need to upgrade genesis to the latest version (our genesis version is **0.3.3**).  

