import os
import sys
import lerobot.robots.uon_amr.uon_amr
import lerobot.teleoperators.uon_rc_teleoperator.uon_rc_teleoperator
from scripts.lerobot_record import main

if __name__ == "__main__":
    os.environ["RERUN_IP"] = "127.0.0.1:9876"
     
    if len(sys.argv) == 1:
        print("[System] UON AMR 기본 인자로 실행합니다...")
        
        sys.argv.extend([
            "--robot.type=uon_amr",
            "--robot.motor_port=/dev/ttyUSB11",
            "--robot.lidar_port=/dev/ttyUSB12",
            "--robot.lidar_baudrate=460800",
            
            "--teleop.type=uon_rc_teleoperator",
            "--teleop.port=/dev/ttyUSB14",
            
            "--dataset.repo_id=uonro/mobile_bot_formal",
            "--dataset.single_task=Drive forward",
            "--dataset.root=data",
            "--dataset.push_to_hub=false",
            
            "--dataset.fps=15",
            "--dataset.episode_time_s=3000",
            "--dataset.num_episodes=10",
            
            "--display_data=false"
        ])
        
    main()
