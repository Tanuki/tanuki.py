import time
from tempfile import TemporaryDirectory

import pytest

from models.function_example import FunctionExample
from trackers.buffered_logger import BufferedLogger


@pytest.fixture(params=[BufferedLogger])
def logger(request):
    with TemporaryDirectory() as temp_dir:
        global ALIGN_FILE_NAME
        ALIGN_FILE_NAME = temp_dir
        yield request.param()


def test_load_log_align(logger):
    runs = 100000

    start_time = time.time()
    for i in range(runs):
        example = FunctionExample((i,), {}, i * 2)
        logger._log_align(str(i), example)
    elapsed_time = time.time() - start_time

    print(f"Time taken for {logger.__class__.__name__}: {elapsed_time} seconds")

def test_load_log_patch(logger):
    runs = 10000

    start_time = time.time()
    for i in range(runs):
        example = FunctionExample((i,), {}, i * 2)
        logger._log_patch(str(i), example)
    elapsed_time = time.time() - start_time

    print(f"Time taken for {logger.__class__.__name__}: {elapsed_time} seconds")

if __name__ == "__main__":
    test_load_log_patch(BufferedLogger())