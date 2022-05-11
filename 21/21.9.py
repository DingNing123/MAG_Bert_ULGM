# 21.9.py
import logging

logging.basicConfig(level=logging.DEBUG,
                    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt = '%m-%d %H:%M',
                    filename = '/tmp/myapp.log',
                    filemode ='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
console.setFormatter(formatter)

logging.getLogger('').addHandler(console)

logging.info('Jackdaws love my big sphinx of quartz.')

logger1 = logging.getLogger('myapp.area1')
logger2 = logging.getLogger('myapp.area2')

logger1.debug('Quick zephyrs blow, vexing daft Jim.')
logger1.info('How quickly daft jumping zebras vex.')
logger2.warning('Jail zesty vixen who grabbed pay from quck.')
logger2.error('The five boxing wizards jummp quickly.')
