from functools import partial
from datetime import datetime

# level = 'hint'

_LOG_COLORS = {
    'important': '\x1b[1;39;49m',
    'info': '\x1b[0;35;39m',
    'detail': '\x1b[2;34;49m',
    'hint': '\x1b[0;36;49m',
    'warning': '\x1b[1;31;49m',
}

_LOG_LEVELS = ['warning', 'important', 'info', 'hint', 'detail']
_LOG_LEVELS_DICT = dict((l, i) for i, l in enumerate(_LOG_LEVELS))


class Logger(object):

    level = 'hint'

    def _log(self, *args, log_level='info'):
        if _LOG_LEVELS_DICT[self.level] >= _LOG_LEVELS_DICT[log_level]:
            print(_LOG_COLORS[log_level] + datetime.now().strftime('%H:%M:%S'), *args, '\033[0m')

    def __getattribute__(self, name):
           
        if name in _LOG_COLORS:
            return partial(self._log, log_level=name)

        return super().__getattribute__(name)


log = Logger()
__all__ = ['log']
