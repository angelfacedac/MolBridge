from src.utils.add_graph import add_graph
from src.utils.backup import zip_project_files
from src.utils.metrics import metrics
from src.utils.set_seed import set_seed


def backup_(output_filename, exclude_dirs=('data', 'tmp', 'logs', '.git', '.idea')):
    def decorator(func):
        def wrapper(*args, **kwargs):
            zip_project_files(output_filename, exclude_dirs)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def add_graph_(tensorboard_path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            add_graph(tensorboard_path)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def set_seed_(seed):
    def decorator(func):
        def wrapper(*args, **kwargs):
            set_seed(seed)
            return func(*args, **kwargs)
        return wrapper
    return decorator

