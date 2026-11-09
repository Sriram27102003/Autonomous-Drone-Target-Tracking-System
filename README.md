# Autonomous Target Tracking System

A ROS 1 (Noetic) package that makes a TurtleBot3 **find, track and follow a person** in a Gazebo
simulation while it **maps its surroundings with SLAM**.

- **Detection:** YOLOv8 (via `ultralytics`) limited to the *person* class, or OpenCV's HOG
  detector as a lightweight fallback.
- **Tracking:** an IoU tracker gives every person a stable ID across frames, survives short
  occlusions, and filters one-frame false positives.
- **Following:** a proportional controller turns towards the chosen person and holds a set
  distance. Distance comes from the lidar when it agrees with a camera-based estimate,
  otherwise from the camera alone. There's a lidar safety stop, and the robot searches
  when the target is lost.
- **Mapping:** GMapping SLAM, with an obstacle-avoiding explorer that maps hands-free.
- **World:** a forest with trees and two walking people (Gazebo actors), so there's
  something to track.

> The repository is named "Drone", but the current implementation runs on a ground robot.
> The detection, tracking and control code doesn't depend on the robot, and is the part
> that would carry over to a drone (see [Roadmap](#roadmap)).

## How it fits together

```
 /camera/rgb/image_raw ──► person_tracker ──► /person_tracker/target ──► target_follower ──► /cmd_vel
                           (YOLO/HOG + IoU)    /person_tracker/detections        ▲
                                │              /human_detection_alert            │
                                └──► /person_tracker/annotated     /scan, /camera/rgb/camera_info

 /scan ──► explorer ──► /cmd_vel           (mapping mode only)
 /scan, /tf ──► slam_gmapping ──► /map
```

| Node | Subscribes | Publishes |
|---|---|---|
| `person_tracker_node.py` | `image`, `~select_target` (Int32, `-1` = auto) | `~detections` (Detection2DArray), `~target` (Detection2D), `~annotated` (Image), `/human_detection_alert` (String) |
| `target_follower_node.py` | `target`, `camera_info`, `scan` | `cmd_vel`; service `~enable` (SetBool) |
| `explorer_node.py` | `scan` | `cmd_vel`; service `~enable` (SetBool) |

In each detection, `results[0].id` is the person's **track ID**. All parameters are documented
in [`config/`](config).

## Requirements

- Ubuntu 20.04, ROS Noetic, Gazebo 11
- TurtleBot3 packages and `vision_msgs`:
  ```bash
  sudo apt install ros-noetic-turtlebot3 ros-noetic-turtlebot3-simulations \
       ros-noetic-turtlebot3-slam ros-noetic-gmapping ros-noetic-map-server \
       ros-noetic-vision-msgs ros-noetic-image-view
  ```
- YOLOv8 (not needed for `backend:=hog`):
  ```bash
  pip3 install -r requirements.txt
  ```

## Build

```bash
mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src
git clone https://github.com/Sriram27102003/Autonomous-Drone-Target-Tracking-System.git human_detector
cd ~/catkin_ws && rosdep install --from-paths src --ignore-src -y
catkin_make && source devel/setup.bash
export TURTLEBOT3_MODEL=waffle   # waffle or waffle_pi; burger has no camera
```

## Run

**Track and follow a person** (Gazebo, SLAM, RViz and the annotated camera view):
```bash
roslaunch human_detector tracking.launch
```
Useful options: `backend:=hog`, `follow:=false` (detect only), `slam:=false`, `gui:=false`.

**Map the forest autonomously** (people are reported on `/human_detection_alert` along the way):
```bash
roslaunch human_detector mapping.launch
```
To drive by hand, use `explore:=false` and run `roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch`
in a second terminal. Save the map with:
```bash
roslaunch human_detector save_map.launch map_name:=forest_map
```

**At runtime:**
```bash
rostopic pub -1 /person_tracker/select_target std_msgs/Int32 2     # follow person #2
rostopic pub -1 /person_tracker/select_target std_msgs/Int32 -- -1 # back to automatic
rosservice call /target_follower/enable false                      # stop following
rostopic echo /human_detection_alert
```

## Tests

The tracking and control logic is plain Python, so the unit tests run without ROS:
```bash
python3 -m unittest discover -s test -v
```
or with catkin: `catkin_make run_tests_human_detector`.

## Notes and limitations

- Gazebo actors (the walking people) have no collision, so the lidar doesn't see them.
  The follower falls back to estimating distance from the person's box height (assuming
  someone 1.7 m tall).
- YOLOv8n runs at a few frames per second on CPU. Set `device: "0"` in
  `config/tracker.yaml` if you have a CUDA GPU.
- The HOG backend needs OpenCV 4.x (the version ROS Noetic ships), because OpenCV 5
  removed `HOGDescriptor` from its core module.

## Roadmap

- Re-identification by appearance (e.g. Deep SORT / ByteTrack embeddings) to keep IDs
  through long occlusions
- Drone version: PX4 SITL + MAVROS, sending velocity setpoints instead of `cmd_vel`
- Port to ROS 2 (Humble/Jazzy). ROS 1 Noetic reached end of life in May 2025.

## Project layout

```
config/    tracker / follower / explorer parameters
launch/    simulation, tracking, mapping, save_map
scripts/   ROS nodes
src/human_detector/   detectors, tracker, control (no ROS dependencies)
test/      unit tests
worlds/    forest_people.world
```

## License

MIT. See [LICENSE](LICENSE). Contact: sriram.surya@icloud.com
