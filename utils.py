import logging
import time
import numpy as np

def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("./logs/log_{}_N_{:.3f}.txt".format(time.strftime("%Y-%m-%d %H-%M-%S"), np.random.uniform()))
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger