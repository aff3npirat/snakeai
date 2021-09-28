from pathlib import Path
root_dir = Path(__path__[0][:-len(__name__) - 1])
from .main import train
