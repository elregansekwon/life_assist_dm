# life_assist_dm

## Install

```jsx
cd ~/ros_ws && mkdir dm_ws/src
cd ~/ros_ws/dm_ws/src
git clone https://github.com/keti-ai/life_assist_dm.git

cd life_assist_dm/life_assist_dm
pip install -r requirements.txt
cd ../../..

colcon build --symlink-install
```

## Usage

```jsx
# Launch
ros2 launch life_assist_dm dialog_manager.launch.py
```