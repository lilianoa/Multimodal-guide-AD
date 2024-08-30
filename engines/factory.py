from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from engines.engine import Engine, KDEngine
def factory():
    opt = Options()['engine']
    if opt['name'] == 'KDEngine':
        engine = KDEngine()
    else:
        engine = Engine()
    return engine