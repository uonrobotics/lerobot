import os
import sys
import lerobot.robots.uon_amr.uon_amr
import lerobot.teleoperators.uon_rc_teleoperator.uon_rc_teleoperator
from scripts.lerobot_record import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("[System] UON AMR 기본 인자로 실행합니다...")
        
        sys.argv.extend([
            "--robot.type=uon_amr",
            "--robot.motor_port=/dev/ttyUSB11",
            "--robot.lidar_port=/dev/ttyUSB12",
            "--robot.lidar_baudrate=460800",
            
            "--teleop.type=uon_rc_teleoperator",
            "--teleop.port=/dev/ttyUSB14",
            
            "--dataset.repo_id=uonro/goto1",
            "--dataset.single_task=Search right to find marker 1, approach it, and stop.",
            "--dataset.root=data_goto1",
            # "--resume=true",  # only when resuming 
            "--dataset.push_to_hub=false",
            
            "--dataset.fps=15",
            "--dataset.episode_time_s=3000",
            "--dataset.num_episodes=10",
            
            "--display_data=false"
        ])
        
    main()
