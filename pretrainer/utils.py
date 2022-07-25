from logging import getLogger, StreamHandler, Formatter, INFO, DEBUG, WARN, WARNING, ERROR, CRITICAL


def get_logger(name, log_level=INFO):
    logger = getLogger(name)
    logger.setLevel(log_level)
    handler = StreamHandler()
    handler.setFormatter(Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.propaget = False
    return logger

