import logging

logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)

logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')
