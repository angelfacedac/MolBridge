import warnings
import faulthandler
import multiprocessing

from src.load_config import CONFIG
from src.run import run, run_pyg

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics._classification')
faulthandler.enable()


def main():
    if CONFIG['is_pyg']:
        run_pyg()
    else:
        run()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

