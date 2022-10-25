import os
import logging


def get_logger(log_dir, log_name):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 1. create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # 2. create log
    log_name = os.path.join(log_dir, '{}.info.log'.format(log_name))

    # 3. setting output formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 4. log file processor
    fh = logging.FileHandler(log_name)
    fh.setFormatter(formatter)

    # 5. setting screen stdout output processor
    sh = logging.StreamHandler(stream=None)
    sh.setFormatter(formatter)

    # 6. add the processor to the logger
    logger.addHandler(fh)  # add file
    logger.addHandler(sh)  # add sh

    return logger
