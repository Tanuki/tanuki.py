import os
from typing import Literal

from appdirs import user_data_dir

from monkey_patch.persistence.filter.filesystem_bloom import FileSystemBloomFilterPersistence
from monkey_patch.trackers.abc_buffered_logger import ABCBufferedLogger, ENVVAR, \
    LIB_NAME, ALIGN_FILE_EXTENSION, PATCH_FILE_EXTENSION


class FilesystemBufferedLogger(ABCBufferedLogger):
    def __init__(self, name, level=15):
        self.log_directory = self._get_log_directory()
        super().__init__(name, level)

    def get_bloom_filter_persistence(self):
        return FileSystemBloomFilterPersistence(log_directory=self.log_directory)

    def get_patch_location_for_function(self, func_hash, extension=""):
        return os.path.join(self.log_directory, func_hash + extension)

    def ensure_persistence_location_exists(self):
        log_directory = self.log_directory
        # Create the folder if it doesn't exist
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

    def _get_log_directory(self):
        filename = "functions"
        env_dir = os.getenv(ENVVAR)
        if env_dir and os.path.isdir(env_dir):
            return os.path.join(env_dir, filename)

        library_dir = os.path.join(user_data_dir(LIB_NAME), filename)
        if os.path.isdir(library_dir) or not os.path.exists(library_dir):
            return library_dir

        current_dir = os.getcwd()
        while current_dir != os.path.root:
            if ".git" in os.listdir(current_dir):
                return os.path.join(current_dir, filename)
            current_dir = os.path.dirname(current_dir)

        return os.path.join(os.getcwd(), filename)

    def load_dataset(self, dataset_type, func_hash, return_type="both"):
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

    def load_existing_datasets(self):
        log_directory = self.log_directory
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

    def write(self, path: str, data: str, mode: Literal["w", "a", "a+b"] = "w") -> None:
        with open(path, mode) as f:
            f.write(data)

    def read(self, path: str) -> str:
        with open(path, "r") as f:
            return f.read()

    def get_hash_from_path(self, path) -> str:
        return path.replace(PATCH_FILE_EXTENSION, "").\
            replace(self.log_directory, "").\
            lstrip("/").\
            lstrip("\\")

    def does_object_exist(self, path) -> bool:
        return os.path.exists(path)