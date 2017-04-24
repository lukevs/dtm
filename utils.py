"""module for data utils"""

import os
import json
import logging.config


LOGGING_CONFIG_PATH = 'logging.json'


def setup_logging(path=LOGGING_CONFIG_PATH):
    """sets up logging"""

    default_level=logging.INFO
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
            logging.config.dictConfig(config)


def read_data(path):
    """reads data from the path

    :param path: path to data
    :type path: str

    :returns: data
    :rtype: dict
    """

    with open(path, 'r') as f:
        return json.load(f)
