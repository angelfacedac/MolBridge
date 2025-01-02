import warnings

from src.run import run
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics._classification')

run()
