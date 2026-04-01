import json
from multiprocessing import Process, Value
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ROOT_PATH = Path.home() / '.cache/huggingface/lerobot'


def create_lerobot_dataset(repo_id: str, features: dict, fps: int, robot_type: str, mode: str = 'interactive') -> LeRobotDataset | None:
    """
    기존 데이터셋의 피쳐 구성을 확인하고, 지정된 모드 또는 사용자 선택에 따라 로드하거나 새로 생성함
    mode:
        - 'interactive': (기본값) 기존처럼 input()으로 사용자에게 묻습니다.
        - 'resume': 호환되면 무조건 이어서 작성합니다.
        - 'overwrite': 기존 데이터를 강제로 삭제하고 새로 생성합니다.
    """
    dataset_path = ROOT_PATH / repo_id
    info_path = dataset_path / 'meta' / 'info.json'

    should_create_new = True
    dataset = None

    if dataset_path.exists():
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    existing_info = json.load(f)

                existing_features = existing_info.get('features', {})

                # 피쳐 호환성 체크
                is_compatible = True
                for key, expected_val in features.items():
                    if key not in existing_features:
                        print(f'[Warn ] 필수 피쳐 누락: {key}')
                        is_compatible = False
                        break

                    existing_shape = existing_features[key].get('shape')
                    if existing_shape != list(expected_val['shape']):
                        print(f'[Warn ] 피쳐 형태 불일치 ({key}): {existing_shape} vs {list(expected_val["shape"])}')
                        is_compatible = False
                        break

                if is_compatible:
                    print(f'\n\n')
                    print(f'[Info ] 기존 데이터셋({repo_id})의 피쳐 구성이 현재 설정과 동일합니다.')

                    if mode == 'interactive':
                        user_input = input(f'[Query] 기존 데이터셋을 불러올까요? (y: 불러오기 / n: 삭제 후 새로 생성): ').lower()
                    else:
                        user_input = 'y' if mode == 'resume' else 'n'
                        print(f'[Info ] 설정된 모드({mode})에 따라 자동으로 {"불러오기" if user_input == "y" else "삭제 후 생성"}를 진행합니다.')

                    if user_input == 'y':
                        should_create_new = False
                        # 기존 데이터셋 로드
                        dataset = LeRobotDataset(repo_id=repo_id)
                        # 이미지 라이터 시작
                        dataset.start_image_writer(num_processes=4, num_threads=8)
                        print(f'[Info ] 기존 데이터셋을 성공적으로 불러왔습니다. (에피소드 수: {dataset.num_episodes})')
                else:
                    print(f'\n\n')
                    print(f'[Warn ] 기존 데이터셋의 피쳐 구성이 현재 설정과 다릅니다.')

                    if mode == 'interactive':
                        user_input = input(f'[Query] 기존 데이터셋을 삭제하고 새로 생성할까요? (y/n): ').lower()
                    else:
                        user_input = 'y' if mode == 'overwrite' else 'n'
                        print(f'[Info ] 설정된 모드({mode})에 따라 자동으로 {"삭제 후 생성" if user_input == "y" else "중단"}을 진행합니다.')

                    if user_input != 'y':
                        print(f'[Info ] 작업을 중단합니다.')
                        return None
            except Exception as e:
                # 404 에러 등 서버 관련 에러가 발생하면 여기서 잡힙니다.
                print(f'\n\n')
                print(f'[Error] 데이터셋 처리 중 오류 발생: {e}')
                print(f'[Info ] 서버 연결 문제일 수 있습니다. 로컬 데이터를 삭제하고 새로 생성하시겠습니까?')

                if mode == 'interactive':
                    user_input = input(f'[Query] 기존 데이터 삭제 후 새로 생성 (y/n): ').lower()
                else:
                    user_input = 'y' if mode == 'overwrite' else 'n'
                    print(f'[Info ] 설정된 모드({mode})에 따라 자동으로 {"삭제 후 생성" if user_input == "y" else "중단"}을 진행합니다.')

                if user_input != 'y':
                    return None

        if should_create_new:
            import shutil
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
                print(f'[Info ] 기존 데이터셋 디렉토리를 제거했습니다: {dataset_path}')

    if should_create_new:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            robot_type=robot_type,
            use_videos=True,
            image_writer_processes=4,
            image_writer_threads=8,
        )
        print(f'[Info ] 새 데이터셋이 생성되었습니다: {dataset_path}')

    return dataset


def dataset_worker(queue, repo_id, features, fps, robot_type, mode, is_saving_val):
    dataset = create_lerobot_dataset(repo_id=repo_id,
                                     features=features,
                                     fps=fps,
                                     robot_type=robot_type,
                                     mode=mode)
    if dataset is None:
        return

    while True:
        try:
            msg = queue.get()
            if msg is None: break

            cmd = msg.get('cmd')

            if cmd == 'add_frame':
                dataset.add_frame(msg['data'])
            elif cmd == 'save':
                # 저장 시작: True (1)
                is_saving_val.value = 1
                print(f'\n[Dataset Process] 에피소드 저장 중...')
                dataset.save_episode()
                # 저장 완료: False (0)
                is_saving_val.value = 0
                print(f'[Dataset Process] 에피소드 저장 완료')
            elif cmd == 'cancel':
                dataset.clear_episode_buffer()
                print(f'[Info ] [Dataset Process] 에피소드 버퍼 초기화 완료.')
            elif cmd == 'done':
                dataset.finalize()
                print(f'[Info ] [Dataset Process] 데이터셋 저장 완료!')
                break
        except Exception as e:
            print(f"[Dataset Process Error] {e}")
            is_saving_val.value = 0 # 에러 발생 시 상태 초기화


def start_dataset_worker(queue, repo_id, features, fps, dataset_mode, is_saving_val):
    dataset_proc = Process(target=dataset_worker, args=(queue, repo_id, features, fps, 'omy_f3m', dataset_mode, is_saving_val))
    dataset_proc.daemon = False
    dataset_proc.start()