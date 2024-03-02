import os

AIRGYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
AIRGYM_ENVS_DIR = os.path.join(AIRGYM_ROOT_DIR, 'aerial_gym', 'envs')

print("AIRGYM_ROOT_DIR", AIRGYM_ROOT_DIR)