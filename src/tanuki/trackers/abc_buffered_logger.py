import json
from abc import abstractmethod
from typing import Dict, Any, Literal

from tanuki.bloom_filter import BloomFilter
from tanuki.constants import EXPECTED_ITEMS, FALSE_POSITIVE_RATE, ALIGN_FILE_EXTENSION, \
    POSITIVE_FILE_EXTENSION, NEGATIVE_FILE_EXTENSION, PATCH_FILE_EXTENSION
from tanuki.persistence.filter.bloom_interface import IBloomFilterPersistence
from tanuki.trackers.dataset_worker import DatasetWorker
from tanuki.models.function_config import FunctionConfig

# PATCH_FILE_EXTENSION_TYPE = Literal[".patches"]
# ALIGN_FILE_EXTENSION_TYPE = Literal[".alignments"]
# POSITIVE_EMBEDDING_FILE_EXTENSION_TYPE = Literal[".positive_embedding"]
# NEGATIVE_EMBEDDING_FILE_EXTENSION_TYPE = Literal[".negative_embedding"]
#
# PATCH_FILE_EXTENSION: PATCH_FILE_EXTENSION_TYPE = ".patches"
# ALIGN_FILE_EXTENSION: ALIGN_FILE_EXTENSION_TYPE = ".alignments"
# POSITIVE_EMBEDDING_FILE_EXTENSION: POSITIVE_EMBEDDING_FILE_EXTENSION_TYPE = ".contrastive_positives"
# NEGATIVE_EMBEDDING_FILE_EXTENSION: NEGATIVE_EMBEDDING_FILE_EXTENSION_TYPE = ".contrastive_negatives"
#
# EXPECTED_ITEMS = 10000
# FALSE_POSITIVE_RATE = 0.01
# LIB_NAME = "tanuki"
# ENVVAR = "TANUKI_LOG_DIR"


class ABCBufferedLogger(DatasetWorker):
    def __init__(self, name, level=15):
        self.buffers = {}
        self.mapped_files = {}
        self.miss_count = 0
        self.hit_count = 0
        self.flush_limit = {}
        self.buffer_rolling_size = {}
        self.write_count = 0
        self.write_limit = 1000  # Save the Bloom filter every 1000 writes

        super().__init__(name, level)
        self.bloom_filter = self.create_bloom_filter()
        self.load_bloom_filter()

        self.default_function_config = FunctionConfig()

    @abstractmethod
    def get_bloom_filter_persistence(self) -> IBloomFilterPersistence:
        """
        Get an instance of the bloom filter persistence provider. This exposes some persistent file storage,
        that must support reading and writing raw byte streams.
        :return:
        """
        pass

    @abstractmethod
    def load_existing_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the lengths of all datasets backing the registered functions, including aligns.
        :return:
        """
        pass

    @abstractmethod
    def ensure_persistence_location_exists(self):
        """
        Ensure that the place we will be writing to actually exists. If not, create it.
        """
        pass

    @abstractmethod
    def get_patch_location_for_function(self, func_hash, extension="") -> str:
        """
        Get the address of the function patch file.
        :param func_hash: The representation of the function
        :param extension: Whether this is a patch or an alignment
        :return:
        """
        pass

    @abstractmethod
    def write(self, path, data, mode="a") -> None:
        pass

    @abstractmethod
    def read(self, path) -> str:
        pass

    @abstractmethod
    def get_hash_from_path(self, path) -> str:
        pass

    @abstractmethod
    def does_object_exist(self, path) -> bool:
        pass

    def create_bloom_filter(self):
        bloom_filter_persistence = self.get_bloom_filter_persistence()

        bloom_filter = BloomFilter(
            bloom_filter_persistence,
            expected_number_of_elements=EXPECTED_ITEMS,
            false_positive_probability=FALSE_POSITIVE_RATE)
        return bloom_filter

    def load_bloom_filter(self):
        try:
            self.bloom_filter.load()
        except FileNotFoundError:
            self.debug("No Bloom filter found. Creating a new one.")

    def write_symbolic_align_call(self, func_hash, example) -> bool:
        log_file_path = self.get_patch_location_for_function(func_hash, extension=ALIGN_FILE_EXTENSION)
        try:
            # Now, write to the file
            dumpable_object = str(example.__dict__)
            self.write(log_file_path, dumpable_object + "\n", mode="a")
            return True
        except Exception as e:
            return False

    def write_embeddable_align_call(self, func_hash, example, positive=True) -> bool:
        if positive:
            log_file_path = self.get_patch_location_for_function(func_hash, extension=POSITIVE_FILE_EXTENSION)
        else:
            log_file_path = self.get_patch_location_for_function(func_hash, extension=NEGATIVE_FILE_EXTENSION)

        try:
            # Now, write to the file
            dumpable_object = str(example.__dict__)
            self.write(log_file_path, dumpable_object + "\n", mode="a")
            return True
        except Exception as e:
            return False

    def log_embeddable_align(self, func_hash, example, positive=True, **kws):
        """
        Log a contrastive function invocation
        Args:
            func_hash: A string representation of the function signature and input parameters
            example: The example object
            positive: Whether the example is positive or negative
            **kws:
        """
        successfully_saved, new_datapoint = False, False
        try:
            self.ensure_persistence_location_exists()
        except Exception as e:
            return successfully_saved, new_datapoint

        # prepend the function hash to the example
        bloom_filter_representation = func_hash + '_' + str(example.__dict__) + '\n'
        # Check Bloom Filter
        if self.bloom_filter.lookup(bloom_filter_representation):
            return successfully_saved, new_datapoint
        new_datapoint = True
        # add to bloom filter
        self.bloom_filter.add(bloom_filter_representation)
        self.save_bloom_filter()

        successfully_saved = self.write_embeddable_align_call(func_hash, example, positive)
        return successfully_saved, new_datapoint

    def log_symbolic_align(self, func_hash, *args, **kws):
        """
        Log an align function invocation to the file system
        :param func_hash: A string representation of the function signature and input parameters
        :param args: Example objects
        :param kws:
        :return:
        """
        successfully_saved, new_datapoint = False, False
        try:
            self.ensure_persistence_location_exists()
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

        successfully_saved = self.write_symbolic_align_call(func_hash, example)
        return successfully_saved, new_datapoint

    def log_symbolic_patch(self, func_hash, example):
        """
        Log a patched function invocation to the file system
        :param func_hash: A string representation of the function signature and input parameters
        :param example:
        :return:
        """
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
            self.ensure_persistence_location_exists()
        except Exception as e:
            return {}

        log_file_path = self.get_patch_location_for_function(func_hash, extension=PATCH_FILE_EXTENSION)

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
                self.write(log_file_path, self.buffers[log_file_path], mode="a+b")
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
            self.bloom_filter.save()
        except Exception as e:
            self.warning("Could not save Bloom filter: {}".format(e))

    def flush(self):
        # get log directory
        written_datapoints = {}
        for log_file_path, buffer in self.buffers.items():
            if len(buffer) > 0:
                try:
                    self.write(log_file_path, buffer, mode="a+b")
                    written_datapoints[self.get_hash_from_path(log_file_path)] = self.buffer_rolling_size[log_file_path]
                    self.buffer_rolling_size[log_file_path] = 0
                    buffer.clear()
                except Exception as e:
                    pass
        return written_datapoints

    def load_function_config(self, func_hash):

        """
        Get the config file for the function. Uses the message and log directory
        Config file has to be in .json
        """
        default = False
        try:  # try to get the config from the disk. If inaccessible, create a new default one

            self.ensure_persistence_location_exists()
            log_file_path = self.get_patch_location_for_function(func_hash)

            config_path = f"{log_file_path}.json"
            if not self.does_object_exist(config_path):
                function_config = self.default_function_config
                default = True
                func_config_dict = function_config.to_dict()
                # remove teacher_models from the config 
                func_config_dict.pop("teacher_models")
                self.write_json(config_path, func_config_dict)
            else:
                function_config = FunctionConfig().load_from_dict(self.read_json(config_path))

        except Exception as e:
            function_config = self.default_function_config
            default = True
        return function_config, default

    def update_function_config(self, func_hash, config_to_be_saved):
        """
        Save the config file
        """
        log_file_path = self.get_patch_location_for_function(func_hash)
        config_path = f"{log_file_path}.json"
        try:
            func_config_dict = config_to_be_saved.to_dict()
            # remove teacher_models from the config 
            func_config_dict.pop("teacher_models")
            self.write_json(config_path, func_config_dict)
        except Exception as e:
            pass

    def write_json(self, path, data):
        self.write(path, json.dumps(data))

    def read_json(self, path):
        return json.loads(self.read(path))
