import warnings
import faulthandler

from src.load_config import CONFIG
from src.run import run, run_pyg

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics._classification')
faulthandler.enable()

if CONFIG['is_pyg']:
    run_pyg()
else:
    run()
