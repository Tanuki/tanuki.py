import os
from logging import Logger
import json
from appdirs import user_data_dir

from bloom_filter import BloomFilter, optimal_bloom_filter_params
import openai
import ast
import datetime
from utils import approximate_token_count
from trackers.dataset_worker import DatasetWorker

PATCH_FILE_EXTENSION = ".patches"
ALIGN_FILE_EXTENSION = ".alignments"
EXPECTED_ITEMS = 10000
FALSE_POSITIVE_RATE = 0.01
LIB_NAME = "monkey-patch"
ENVVAR = "MONKEY_PATCH_LOG_DIR"

class BufferedLogger(DatasetWorker):
    def __init__(self, name, level=15):
        self.buffers = {}
        self.mapped_files = {}
        self.miss_count = 0
        self.hit_count = 0
        self.flush_limit = {}
        self.log_directory = self._get_log_directory()
        self.flush_limit = {}
        self.bloom_filter = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))
        self.buffer_rolling_size = {}
        try:
            self.bloom_filter.load(self.log_directory)
        except FileNotFoundError:
            self.bloom_filter = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))

        self.write_count = 0
        self.write_limit = 1000  # Save the Bloom filter every 1000 writes

        super().__init__(name, level)

    def _get_log_directory(self):

        filename = "functions"

        # Check for an environment variable to determine where to write.
        env_dir = os.getenv(ENVVAR)
        if env_dir and os.path.isdir(env_dir):
            return os.path.join(env_dir, filename)

        # Write in a library-specific data directory.
        library_dir = os.path.join(user_data_dir(LIB_NAME), filename)
        if os.path.isdir(library_dir) or not os.path.exists(library_dir):
            return library_dir

        # Determine the root of the project and write there.
        current_dir = os.getcwd()
        while current_dir != os.path.root:
            if ".git" in os.listdir(current_dir):
                return os.path.join(current_dir, filename)
            current_dir = os.path.dirname(current_dir)

        # 4. Write to where the code is being executed from.
        return os.path.join(os.getcwd(), filename)

    def _load_dataset_sizes(self):
        """
        Get all dataset sizes for existing datasets
    
        """

        log_directory = self._get_log_directory()
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        # get all the files in the log directory
        files = os.listdir(log_directory)
        # discard all .json files
        files = [x for x in files if ".json" not in x]
        dataset_lengths = {"alignments": {}, "patches": {}}
        for file in files:
            if ALIGN_FILE_EXTENSION not in file and PATCH_FILE_EXTENSION not in file:
                continue
            elif ALIGN_FILE_EXTENSION in file:
                dataset_type = "alignments"
            else:
                dataset_type = "patches"
            func_hash = file.replace(ALIGN_FILE_EXTENSION, "").replace(PATCH_FILE_EXTENSION, "")
            with open(os.path.join(log_directory, file), "rb") as f:
                try:
                    dataset = f.read().decode('utf-8')
                except UnicodeDecodeError:
                    dataset_lengths[dataset_type][func_hash] = 0
                    continue

            dataset = repr(dataset)
            dataset_lengths[dataset_type][func_hash] = dataset.count("\\n") - dataset.count("\\\\n")

        return dataset_lengths


    def log_align(self, func_hash, *args, **kws):
        log_directory = self._get_log_directory()
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        example = args[0]

        # prepend the function hash to the example
        bloom_filter_representation = func_hash + '_' + str(example.__dict__) + '\n'
        # Check Bloom Filter
        if self.bloom_filter.lookup(bloom_filter_representation):
            return False
        # add to bloom filter
        self.bloom_filter.add(bloom_filter_representation)

        # Create the folder if it doesn't exist
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        log_file_path = os.path.join(log_directory, func_hash+ALIGN_FILE_EXTENSION)

        # Now, write to the file
        with open(log_file_path, "a") as f:
            f.write(str(example.__dict__) + "\n")


    def load_alignments(self):
        """
        Load alignments from persistent storage into memory for faster access.
        """
        align_buffers = {}

        log_directory = self._get_log_directory()
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        # get all the files in the log directory
        files = os.listdir(log_directory)
        # discard all non align files
        files = [x for x in files if ALIGN_FILE_EXTENSION in x]
        for file in files:
            func_hash = file.replace(ALIGN_FILE_EXTENSION, "")
            log_file_path = os.path.join(log_directory, file)
            with open(log_file_path, "rb") as f:
                try:
                    align_buffers[func_hash] = bytearray(f.read())
                except UnicodeDecodeError:
                    align_buffers[func_hash] = bytearray()
                    continue
        return align_buffers


    def log_patch(self, func_hash, example):

        if not isinstance(func_hash, str):
            func_hash = str(func_hash)

        example_data = str(example.__dict__).encode('utf-8') + b'\n'

        bloom_filter_representation = func_hash + '_' + example_data.decode('utf-8')
        # Check Bloom Filter
        if self.bloom_filter.lookup(bloom_filter_representation):
            self.hit_count += 1
            return False

        self.miss_count += 1
        # Add to Bloom Filter
        self.bloom_filter.add(bloom_filter_representation)

        log_directory = self._get_log_directory()
        path = os.path.join(log_directory, func_hash)
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        if os.path.exists(log_directory):
            pass

        log_file_path = os.path.join(log_directory, func_hash+PATCH_FILE_EXTENSION)

        if log_file_path not in self.buffers:
            self.buffers[log_file_path] = bytearray()

        if log_file_path not in self.flush_limit:
            self.flush_limit[log_file_path] = 1

        self.buffers[log_file_path].extend(example_data)

        self.write_count += 1
        if log_file_path not in self.buffer_rolling_size:
            self.buffer_rolling_size[log_file_path] = 1
        else:
            self.buffer_rolling_size[log_file_path] += 1
        if self.write_count >= self.write_limit:
            written_datapoints = self.flush()
            self.save_bloom_filter()
            self.write_count = 0  # Reset counter
            return written_datapoints
        
        if len(self.buffers[log_file_path]) >= min(self.flush_limit[log_file_path], 4096):  # Flush after reaching 4KB
            try:
                written_datapoints = {}
                with open(log_file_path, "a+b") as f:
                    f.write(self.buffers[log_file_path])
                written_datapoints[func_hash] = self.buffer_rolling_size[log_file_path]
            except Exception as e:
                print(f"Error writing to file: {e}")
            self.buffers[log_file_path].clear()
            self.buffer_rolling_size[log_file_path] = 0

            self.flush_limit[log_file_path] = 2 * self.flush_limit[log_file_path]
            return written_datapoints
        return {}

    def save_bloom_filter(self):
        self.bloom_filter.save(self.log_directory)

    def flush(self):
        # get log directory
        log_directory = self._get_log_directory()
        written_datapoints = {}
        for log_file_path, buffer in self.buffers.items():
            if len(buffer) > 0:
                with open(log_file_path, "a+b") as f:
                    f.write(buffer)
                func_hash = log_file_path.replace(PATCH_FILE_EXTENSION, "").replace(log_directory, "").lstrip("/").lstrip("\\")
                written_datapoints[func_hash] = self.buffer_rolling_size[log_file_path]
                self.buffer_rolling_size[log_file_path] = 0
                buffer.clear()
        return written_datapoints
                

    def _load_function_config(self, func_hash):

        """
        Get the config file for the function. Uses the message and log directory
        Config file has to have to be in .json
        """
        log_directory = self._get_log_directory()
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        
        log_file_path = os.path.join(log_directory, func_hash)
        config_path = f"{log_file_path}.json"
        if not os.path.exists(config_path):
            function_config = {"distilled_model": "",
                                           "current_model_stats": {
                                               "trained_on_datapoints": 0,
                                               "running_faults": []},
                                           "last_training_run": {"trained_on_datapoints": 0},
                                           "current_training_run": {},
                                           "teacher_models": ["gpt-4","gpt-4-32k"], # currently supported teacher models
                                           "nr_of_training_runs": 0}

            with open(config_path, "w") as f:
                json.dump(function_config, f)
        else:
            with open(config_path, "r") as f:
                function_config = json.load(f)
        return function_config

    def load_datasets(self, func_hash):
        """
        Load the datasets for a function hash
        """
        log_directory = self._get_log_directory()
        log_file_path = os.path.join(log_directory, func_hash)
        if not os.path.exists(log_file_path+ALIGN_FILE_EXTENSION):
            align_dataset = ""
        else:
            # read in the dataset file
            with open(log_file_path+ALIGN_FILE_EXTENSION, "rb") as f:
                align_dataset = f.read().decode('utf-8')
        
        if not os.path.exists(log_file_path+PATCH_FILE_EXTENSION):
            patch_dataset = ""
        else:
            with open(log_file_path+PATCH_FILE_EXTENSION, "rb") as f:
                patch_dataset = f.read().decode('utf-8')
        return align_dataset, patch_dataset


    def _update_function_config(self, func_hash, config_to_be_saved):
        """
        Save the config file
        """
        log_directory = self._get_log_directory()
        log_file_path = os.path.join(log_directory, func_hash)
        with open(f"{log_file_path}.json", "w") as f:
            json.dump(config_to_be_saved, f)
