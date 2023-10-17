import time

from models.function_example import FunctionExample


class BaseLogger:
    def _log_align(self, message, example):
        pass

    def _log_patch(self, message, example):
        pass

    def load_test(self, runs=10000):
        start_time = time.time()
        for i in range(runs):
            example = FunctionExample((i,), {}, i*2)
            self._log_align(str(i), example)
        return time.time() - start_time