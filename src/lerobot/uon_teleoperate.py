import sys
import draccus

# 1. 우리가 만든 모듈 임포트
# (파일 경로: src/lerobot/robots/uon_amr/uon_amr.py)
import lerobot.robots.uon_amr.uon_amr as uon_robot_module
import lerobot.teleoperators.uon_rc_teleoperator.uon_rc_teleoperator as uon_teleop_module

# 2. LeRobot의 기본 설정 클래스 가져오기
from lerobot.robots import RobotConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.scripts.lerobot_teleoperate import main

def force_register_configs():
    """
    draccus 레지스트리에 우리가 만든 설정을 강제로 등록하는 함수
    """
    # 1. 로봇 등록 ('uon_amr' 이름으로 UONAMRConfig 클래스 연결)
    known_robots = RobotConfig.get_known_choices()
    if "uon_amr" not in known_robots:
        print(f"[System] 'uon_amr' 로봇 설정이 레지스트리에 없어서 강제 등록합니다.")
        RobotConfig.register_subclass("uon_amr", uon_robot_module.UONAMRConfig)
    else:
        print(f"[System] 'uon_amr' 로봇 설정이 정상적으로 인식되었습니다.")

    # 2. 조종기 등록 ('uon_rc_teleoperator' 이름으로 Config 클래스 연결)
    known_teleops = TeleoperatorConfig.get_known_choices()
    if "uon_rc_teleoperator" not in known_teleops:
        print(f"[System] 'uon_rc_teleoperator' 조종기 설정이 레지스트리에 없어서 강제 등록합니다.")
        TeleoperatorConfig.register_subclass("uon_rc_teleoperator", uon_teleop_module.UONRcTeleoperatorConfig)
    else:
        print(f"[System] 'uon_rc_teleoperator' 조종기 설정이 정상적으로 인식되었습니다.")

if __name__ == "__main__":
    # 강제 등록 실행
    force_register_configs()

    # 터미널 인자 자동 채우기
    if len(sys.argv) == 1:
        print("[System] 기본 인자로 실행합니다...")
        sys.argv.extend([
            "--robot.type=uon_amr",
            "--robot.motor_port=/dev/ttyUSB11",
            "--teleop.type=uon_rc_teleoperator",
            "--teleop.port=/dev/ttyUSB14",
            "--display_data=false"
        ])
    
    # 메인 실행
    main()