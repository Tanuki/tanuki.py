import json
import os

from appdirs import user_data_dir

from monkey_patch.bloom_filter import BloomFilter, optimal_bloom_filter_params
from monkey_patch.language_models.language_modeler import ApiModelFactory
from monkey_patch.trackers.dataset_worker import DatasetWorker
import json

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

        self.default_function_config = {"distilled_model": ApiModelFactory.get_distilled_model(os.getenv('API_MODEL')),
                                        "current_model_stats": {
                                            "trained_on_datapoints": 0,
                                            "running_faults": []},
                                        "last_training_run": {"trained_on_datapoints": 0},
                                        "current_training_run": {},
                                        "teacher_models": ApiModelFactory.get_teacher_model(os.getenv('API_MODEL')),
                                        # currently supported teacher models
                                        "nr_of_training_runs": 0}

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

    def _load_dataset(self, dataset_type, func_hash, return_type="both"):
        """
        Get the size of the dataset for a function hash
        """
        log_directory = self._get_log_directory()
        dataset_type_map = {"alignments": ALIGN_FILE_EXTENSION, "patches": PATCH_FILE_EXTENSION}

        log_file_path = os.path.join(log_directory, func_hash + dataset_type_map[dataset_type])
        if not os.path.exists(log_file_path):
            if return_type == "both":
                return 0, None
            elif return_type == "dataset":
                return None
            elif return_type == "length":
                return 0
        try:
            with open(log_file_path, "rb") as f:
                dataset = f.read()
            dataset_string = repr(dataset)
            dataset_length = dataset_string.count("\\n") - dataset_string.count("\\\\n")
            if return_type == "both":
                return dataset_length, dataset
            elif return_type == "dataset":
                return dataset
            elif return_type == "length":
                return dataset_length
        except Exception as e:
            if return_type == "both":
                return 0, None
            elif return_type == "dataset":
                return None
            elif return_type == "length":
                return 0

    def _load_existing_datasets(self):
        log_directory = self._get_log_directory()
        dataset_lengths = {"alignments": {}, "patches": {}}
        try:
            if not os.path.exists(log_directory):
                os.makedirs(log_directory)
            # get all the files in the log directory
            files = os.listdir(log_directory)
            # discard all .json files
            files = [x for x in files if ".json" not in x]
        except Exception as e:
            return dataset_lengths

        for file in files:
            if ALIGN_FILE_EXTENSION not in file and PATCH_FILE_EXTENSION not in file:
                continue
            elif ALIGN_FILE_EXTENSION in file:
                dataset_type = "alignments"
            else:
                dataset_type = "patches"
            func_hash = file.replace(ALIGN_FILE_EXTENSION, "").replace(PATCH_FILE_EXTENSION, "")
            dataset_lengths[dataset_type][func_hash] = -1
        return dataset_lengths

    def log_align(self, func_hash, *args, **kws):
        successfully_saved, new_datapoint = False, False
        try:
            log_directory = self._get_log_directory()
            # Create the folder if it doesn't exist
            if not os.path.exists(log_directory):
                os.makedirs(log_directory)
        except Exception as e:
            return successfully_saved, new_datapoint

        example = args[0]

        # prepend the function hash to the example
        bloom_filter_representation = func_hash + '_' + str(example.__dict__) + '\n'
        # Check Bloom Filter
        if self.bloom_filter.lookup(bloom_filter_representation):
            return successfully_saved, new_datapoint
        new_datapoint = True
        # add to bloom filter
        self.bloom_filter.add(bloom_filter_representation)
        self.save_bloom_filter()

        log_file_path = os.path.join(log_directory, func_hash + ALIGN_FILE_EXTENSION)

        try:
            # Now, write to the file
            dumpable_object = str(example.__dict__)
            with open(log_file_path, "a") as f:
                f.write(dumpable_object + "\r\n")
            successfully_saved = True
        except Exception as e:
            pass
        return successfully_saved, new_datapoint

    def log_patch(self, func_hash, example):

        if not isinstance(func_hash, str):
            func_hash = str(func_hash)

        example_data = str(example.__dict__).encode('utf-8') + b'\n'

        bloom_filter_representation = func_hash + '_' + example_data.decode('utf-8')
        # Check Bloom Filter
        if self.bloom_filter.lookup(bloom_filter_representation):
            self.hit_count += 1
            return {}

        self.miss_count += 1
        # Add to Bloom Filter
        self.bloom_filter.add(bloom_filter_representation)

        try:
            log_directory = self._get_log_directory()
            if not os.path.exists(log_directory):
                os.makedirs(log_directory)

        except Exception as e:
            return {}

        log_file_path = os.path.join(log_directory, func_hash + PATCH_FILE_EXTENSION)

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
            written_datapoints = {}
            try:
                with open(log_file_path, "a+b") as f:
                    f.write(self.buffers[log_file_path])
                # update buffers
                written_datapoints[func_hash] = self.buffer_rolling_size[log_file_path]
                self.buffers[log_file_path].clear()
                self.buffer_rolling_size[log_file_path] = 0
                self.flush_limit[log_file_path] = 2 * self.flush_limit[log_file_path]
                self.save_bloom_filter()
            except Exception as e:
                pass

            return written_datapoints
        return {}

    def save_bloom_filter(self):
        try:
            self.bloom_filter.save(self.log_directory)
        except Exception as e:
            pass

    def flush(self):
        # get log directory
        log_directory = self._get_log_directory()
        written_datapoints = {}
        for log_file_path, buffer in self.buffers.items():
            if len(buffer) > 0:
                try:
                    with open(log_file_path, "a+b") as f:
                        f.write(buffer)
                    func_hash = log_file_path.replace(PATCH_FILE_EXTENSION, "").replace(log_directory, "").lstrip(
                        "/").lstrip("\\")
                    written_datapoints[func_hash] = self.buffer_rolling_size[log_file_path]
                    self.buffer_rolling_size[log_file_path] = 0
                    buffer.clear()
                except Exception as e:
                    pass
        return written_datapoints

    def _load_function_config(self, func_hash):

        """
        Get the config file for the function. Uses the message and log directory
        Config file has to have to be in .json
        """
        default = False
        try:  # try to get the config from the disk. If unacessible, create a new default one
            log_directory = self._get_log_directory()
            if not os.path.exists(log_directory):
                os.makedirs(log_directory)

            log_file_path = os.path.join(log_directory, func_hash)
            config_path = f"{log_file_path}.json"
            if not os.path.exists(config_path):
                function_config = self.default_function_config
                default = True

                with open(config_path, "w") as f:
                    json.dump(function_config, f)
            else:
                with open(config_path, "r") as f:
                    function_config = json.load(f)
        except Exception as e:
            function_config = self.default_function_config
            default = True
        return function_config, default

    def _update_function_config(self, func_hash, config_to_be_saved):
        """
        Save the config file
        """
        log_directory = self._get_log_directory()
        log_file_path = os.path.join(log_directory, func_hash)
        try:
            with open(f"{log_file_path}.json", "w") as f:
                json.dump(config_to_be_saved, f)
        except Exception as e:
            pass
