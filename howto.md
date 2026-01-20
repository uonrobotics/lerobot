# Setup
``` shell
python3.10 -m venv lerobot-env
source lerobot-env/bin/activate
pip install .
pip install rplidar-roboticia
pip3 install pyorbbecsdk2
```

# camera test command
``` shell
ffmpeg -f v4l2 -i /dev/video6 -vframes 1 camera_test_dev_video6.jpg
```

# st10comm test command
``` shell
timeout 5s cat /dev/ttyUSB14 | xxd
```


# using venv 

``` shell
cd ~/workspace/lerobot
source lerobot-env/bin/activate
python3 src/lerobot/uon_teleoperate.py
python3 src/lerobot/uon_record.py

```

# visualize data
``` shell
# AMR PC !! 
python3 src/lerobot/scripts/lerobot_dataset_viz.py     --repo-id uonro/mobile_bot_formal     --root data     --episode-index 0     --save 1     --output-dir visualization_output

# LOCAL PC !! 
cd ~/test 
scp uonro@192.168.27.171:~/workspace/lerobot/lerobot-uon-amr/visualization_output/*.rrd .
```
https://app.rerun.io/

