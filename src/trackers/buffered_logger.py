import os

from bloom_filter import BloomFilter
from monkey import ALIGN_FILE_NAME
from trackers.base_logger import BaseLogger
from utils import optimal_bloom_filter_params

EXPECTED_ITEMS = 5000 # The maximum number of functions we expect to see
FALSE_POSITIVE_RATE = 0.01 # 1% false positive rate


class BufferedLogger(BaseLogger):
    def __init__(self):
        self.buffers = {}
        self.mapped_files = {}

        try:
            self.bloom_filter = self.load_bloom_filter()
        except FileNotFoundError:
            self.bloom_filter = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))

        self.write_count = 0
        self.write_limit = 1000  # Save the Bloom filter every 1000 writes

    def _log_align(self, message, example):
        log_directory = os.path.join(os.getcwd(), ALIGN_FILE_NAME)
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        log_file_path = os.path.join(log_directory, message)
        with open(log_file_path, "a") as f:
            f.write(str(example.__dict__) + "\n")

    def _log_patch(self, message, example):
        example_data = str(example.__dict__).encode('utf-8') + b'\n'

        # Check Bloom Filter
        if self.bloom_filter.lookup(example_data.decode('utf-8')):
            return

        # Add to Bloom Filter
        self.bloom_filter.add(example_data.decode('utf-8'))

        log_directory = os.path.join(os.getcwd(), ALIGN_FILE_NAME)
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        log_file_path = os.path.join(log_directory, message)

        if log_file_path not in self.buffers:
            self.buffers[log_file_path] = bytearray()

        self.buffers[log_file_path].extend(example_data)

        self.write_count += 1

        # TODO: Should we save the Bloom filter every time we flush?
        if self.write_count >= self.write_limit:
            self.flush()
            self.save_bloom_filter()
            self.write_count = 0  # Reset counter

        if len(self.buffers[log_file_path]) >= 4096:  # Flush after reaching 4KB
            with open(log_file_path, "a+b") as f:
                fsize = f.tell()
                f.write(self.buffers[log_file_path])

            self.buffers[log_file_path].clear()

    def save_bloom_filter(self):
        with open('bloom_filter_state.bin', 'wb') as f:
            f.write(self.bloom_filter.bit_array.tobytes())

    def load_bloom_filter(self):
        bloom_filter = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, 0.01))
        with open('bloom_filter_state.bin', 'rb') as f:
            bloom_filter.bit_array.frombytes(f.read())
        return bloom_filter

    def flush(self):
        for log_file_path, buffer in self.buffers.items():
            if len(buffer) > 0:
                with open(log_file_path, "a+b") as f:
                    f.write(buffer)
                buffer.clear()