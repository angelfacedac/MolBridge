import faulthandler
import warnings

from src.utils.test_best_num_workers import test_best_num_workers
from src.utils.vis.t_sne import draw_2d

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics._classification')
faulthandler.enable()

draw_2d()

