#!/usr/bin/env python3

import time
_APP_START_TIME = time.perf_counter()

from modules.profiler import enable_profiler, record_time
enable_profiler()

# Record import time
_import_start = time.perf_counter()
from modules import core
record_time('import_modules', (time.perf_counter() - _import_start) * 1000)

# Record total startup time
record_time('app_startup_to_import', (time.perf_counter() - _APP_START_TIME) * 1000)

if __name__ == '__main__':
    core.run()
