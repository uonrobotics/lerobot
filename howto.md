# Setup
``` shell
python3.10 -m venv lerobot-env
source lerobot-env/bin/activate
pip install -e .
pip install rplidar-roboticia
pip3 install pyorbbecsdk2
pip install flask
```

# camera test command
``` shell
ffmpeg -f v4l2 -i /dev/video6 -vframes 1 camera_test_dev_video6.jpg
```

# st10comm test command
``` shell
timeout 5s cat /dev/ttyUSB14 | xxd
```

# Test teleoperation
``` shell
cd ~/workspace/lerobot
source lerobot-env/bin/activate
python3 src/lerobot/uon_teleoperate.py
python3 src/lerobot/uon_record.py
```

# Collecting data
``` shell
cd ~/workspace/lerobot
source lerobot-env/bin/activate
python3 src/lerobot/uon_record.py
```

# visualize data
``` shell
# AMR PC !! 
for i in {0..{n}}; do # number of episodes
    python3 src/lerobot/scripts/lerobot_dataset_viz.py \
        --repo-id uonro/go_to_1 \
        --root data_goto1 \
        --episode-index $i \
        --save 1 \
        --output-dir visualization_output/episode_$i
done

# one episode
python3 src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id uonro/go_to_1 \
    --root data_goto1 \
    --episode-index 0 \
    --save 1 \
    --output-dir visualization_output/episode_0

# LOCAL PC !! 
cd ~/test 
scp uonro@192.168.27.172:~/workspace/lerobot/visualization_output/*/*.rrd .
```

https://app.rerun.io/

