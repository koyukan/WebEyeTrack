import pathlib

GIT_ROOT = pathlib.Path(__file__).parent.parent.parent
PACKAGE_DIR = GIT_ROOT / 'python' / 'webeyetrack'
DEFAULT_CONFIG = PACKAGE_DIR / 'default_config.yaml'