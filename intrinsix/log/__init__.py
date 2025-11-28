import inspect
import logging
import os


def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 1))


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', -1))

logFormatter = logging.Formatter(fmt='%(asctime)s :: %(name)s :: %(levelname)-5s :: %(message)s')
add_stream_handler_default = False


def init_logger(name: str = None, logging_level=logging.DEBUG, add_stream_handler=None) -> logging.Logger:
    """
    Creates a new logger with the given name
    :param name: The name of the logger
    :param logging_level: The logging level
    :param add_stream_handler: Whether to add a stream handler or not
    :return: The created logger
    """
    if name is None:
        frame = inspect.currentframe()
        frame = frame.f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        name = local_vars["self"].__class__.__name__

    local_rank = get_local_rank()
    if local_rank != -1:
        name = f"{local_rank}_{name}"

    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])

    if add_stream_handler is None:
        add_stream_handler = add_stream_handler_default
    if add_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logFormatter)

        logger.addHandler(stream_handler)

    logger.debug(f"Logger created with name: {name}")
    return logger


def log_to_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logFormatter)

    logging.getLogger().addHandler(file_handler)


import_logger = init_logger("IMPORT")
