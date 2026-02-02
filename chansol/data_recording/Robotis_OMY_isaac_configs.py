from pathlib import Path
DEFAULT_SAVE_ROOT_PATH = Path("/nas/Dataset/VLA/UON/Isaacsim_OMY")
FPS = 25                                           
HF_REPO_ID = "user1/repo1"                     
FEATURES = {
    'observation.images.cam_top': {
        'dtype': 'video',
        'shape': (720, 1280, 3),
        'names': [
            'height',
            'width',
            'channels'
        ]
    },
    'observation.images.cam_wrist': {
        'dtype': 'video',
        'shape': (480, 848, 3),
        'names': [
            'height',
            'width',
            'channels'
        ]
    },
    'observation.state': {
        'dtype': 'float32',
        'shape': (7,),
        'names': [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'rh_r1_joint'
        ]
    },
    'action': {
        'dtype': 'float32',
        'shape': (7,),
        'names': [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'rh_r1_joint'
        ]
    },
}

JOINT_ORDER = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'rh_r1_joint'
]
