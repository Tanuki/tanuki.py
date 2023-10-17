import os
from logging import Logger

from appdirs import user_data_dir

from bloom_filter import BloomFilter, optimal_bloom_filter_params
from models.function_example import FunctionExample

ALIGN_FILE_NAME = "functions"
EXPECTED_ITEMS = 10000
FALSE_POSITIVE_RATE = 0.01
LIB_NAME = "monkey-patch"
ENVVAR = "MONKEY_PATCH_LOG_DIR"


class BufferedLogger(Logger):
    def __init__(self, name, level=15):
        self.buffers = {}
        self.mapped_files = {}
        self.miss_count = 0
        self.hit_count = 0
        self.log_directory = self._get_log_directory()
        self.bloom_filter = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))
        try:
            self.bloom_filter.load(self.log_directory)
        except FileNotFoundError:
            self.bloom_filter = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))

        self.write_count = 0
        self.write_limit = 1000  # Save the Bloom filter every 1000 writes

        super().__init__(self, level)

    def _get_log_directory(self):
        # Check for an environment variable to determine where to write.
        env_dir = os.getenv(ENVVAR)
        if env_dir and os.path.isdir(env_dir):
            return os.path.join(env_dir, ALIGN_FILE_NAME)

        # Write in a library-specific data directory.
        library_dir = os.path.join(user_data_dir(LIB_NAME), ALIGN_FILE_NAME)
        if os.path.isdir(library_dir) or not os.path.exists(library_dir):
            return library_dir

        # Determine the root of the project and write there.
        current_dir = os.getcwd()
        while current_dir != os.path.root:
            if ".git" in os.listdir(current_dir):
                return os.path.join(current_dir, ALIGN_FILE_NAME)
            current_dir = os.path.dirname(current_dir)

        # 4. Write to where the code is being executed from.
        return os.path.join(os.getcwd(), ALIGN_FILE_NAME)

    def log_align(self, message, *args, **kws):
        log_directory = self._get_log_directory()
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        args, kwargs, output = args
        example = FunctionExample(args, kwargs, output)

        log_file_path = os.path.join(log_directory, message)
        with open(log_file_path, "a") as f:
            f.write(str(example.__dict__) + "\n")

    def log_patch(self, message, example):

        if not isinstance(message, str):
            message = str(message)

        example_data = str(example.__dict__).encode('utf-8') + b'\n'

        # Check Bloom Filter
        if self.bloom_filter.lookup(example_data.decode('utf-8')):
            self.hit_count += 1
            return

        self.miss_count += 1
        # Add to Bloom Filter
        self.bloom_filter.add(example_data.decode('utf-8'))

        log_directory = self._get_log_directory()
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        log_file_path = os.path.join(log_directory, message)

        if log_file_path not in self.buffers:
            self.buffers[log_file_path] = bytearray()

        self.buffers[log_file_path].extend(example_data)

        self.write_count += 1

        if self.write_count >= self.write_limit:
            self.flush()
            self.save_bloom_filter()
            self.write_count = 0  # Reset counter

        if len(self.buffers[log_file_path]) >= 4096:  # Flush after reaching 4KB
            with open(log_file_path, "a+b") as f:
                f.write(self.buffers[log_file_path])

            self.buffers[log_file_path].clear()

    def save_bloom_filter(self):
        self.bloom_filter.save(self.log_directory)

    def flush(self):
        for log_file_path, buffer in self.buffers.items():
            if len(buffer) > 0:
                with open(log_file_path, "a+b") as f:
                    f.write(buffer)
                buffer.clear()
