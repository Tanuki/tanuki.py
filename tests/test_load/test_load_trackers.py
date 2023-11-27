import time
from tempfile import TemporaryDirectory

import pytest

from tanuki.models.function_example import FunctionExample
from tanuki.trackers.filesystem_buffered_logger import FilesystemBufferedLogger


@pytest.fixture(params=[FilesystemBufferedLogger])
def logger(request):
    with TemporaryDirectory() as temp_dir:
        global ALIGN_FILE_NAME
        ALIGN_FILE_NAME = temp_dir
        yield request.param("test")


#def test_load_log_align(logger):
#    runs = 100000
#
#    start_time = time.time()
#    for i in range(runs):
#        example = FunctionExample((i,), {}, i * 2)
#        logger.log_align(str(i), example)
#    elapsed_time = time.time() - start_time
#
#    print(f"Time taken for {logger.__class__.__name__}: {elapsed_time} seconds")

#def test_patch_many_functions(logger):
#    runs = 10000
#
#    start_time = time.time()
#    for i in range(runs):
#        example = FunctionExample((i,), {}, i * 2)
#        logger.log_patch(str(i), example)
#    elapsed_time = time.time() - start_time
#
#    print(f"Time taken for {logger.__class__.__name__} to patch {runs} functions: {elapsed_time} seconds")
#
def test_patch_one_function_many_times():
    runs = 100
    logger = FilesystemBufferedLogger("test")
    logger.bloom_filter = logger.create_bloom_filter()
    start_time = time.time()
    for i in range(runs):
        example = FunctionExample((i,), {}, i * 2)
        before_bit_array = logger.bloom_filter.bit_array.copy()

        logger.log_symbolic_patch("test", example)

        after_bit_array = logger.bloom_filter.bit_array
        is_same = before_bit_array == after_bit_array

        assert is_same == False

    logger.save_bloom_filter()
    elapsed_time = time.time() - start_time

    print(f"Time taken for {logger.__class__.__name__} to patch function {runs} times: {elapsed_time} seconds")
    print(f"Hits: {logger.hit_count}, Misses: {logger.miss_count}")

    start_time = time.time()
    for i in range(runs):
        example = FunctionExample((i,), {}, i * 2)

        logger.log_symbolic_patch("test", example)

        after_bit_array = logger.bloom_filter.bit_array
        #is_same = before_bit_array == after_bit_array


    logger.save_bloom_filter()
    elapsed_time = time.time() - start_time

    print(f"Time taken for {logger.__class__.__name__} to patch function {runs} times: {elapsed_time} seconds")
    print(f"Hits: {logger.hit_count}, Misses: {logger.miss_count}")

if __name__ == "__main__":
    #test_patch_many_functions(BufferedLogger())
    test_patch_one_function_many_times()