import pathlib
import os

CWD = pathlib.Path(os.path.abspath(__file__)).parent
TEST_DIR = CWD / 'data'
ROOT_DIR = CWD.parent
MODELS_DIR = ROOT_DIR / 'models'
