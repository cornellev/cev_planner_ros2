### Development Setup

Make a new folder for the ROS2 workspace
and make a `src` folder inside. `cd` into the
`src` folder and then:  
`git clone --recurse-submodules {repo_url}`  
`sudo apt install libnlopt-dev`  
    
Then `cd` back to the workspace root and `colcon build`.

---
### Running
In the workspace root:  
`source install/setup.bash`  
`ros2 launch cev_planner_ros2 launch.py`