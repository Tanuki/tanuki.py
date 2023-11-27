import os
from enum import Enum
from typing import Literal, Union, Optional, Dict

from appdirs import user_data_dir

from tanuki.constants import *
from tanuki.persistence.filter.bloom_interface import IBloomFilterPersistence
from tanuki.persistence.filter.filesystem_bloom import BloomFilterFileSystemDriver
from tanuki.trackers.abc_buffered_logger import ABCBufferedLogger


class FilesystemBufferedLogger(ABCBufferedLogger):
    """
    A class that handles the reading and writing of patch invocations and align statements.
    It includes the logic for a bloom filter, to ensure that we only store unique invocations.
    """

    def __init__(self, name, level=15):
        self.log_directory = self._get_log_directory()
        super().__init__(name, level)

    def get_bloom_filter_persistence(self) -> IBloomFilterPersistence:
        """
        Get an instance of the bloom filter persistence provider. Typically this will be a file system provider.
        :return: A persistence provider
        """
        return BloomFilterFileSystemDriver(log_directory=self.log_directory)

    def get_patch_location_for_function(self, func_hash, extension: Union[
        ALIGN_FILE_EXTENSION_TYPE, PATCH_FILE_EXTENSION_TYPE] = "") -> str:

        """
        Get the local location of the function patch file.
        :param func_hash: The representation of the function
        :param extension: Whether this is a patch or an alignment
        :return:
        """
        return os.path.join(self.log_directory, func_hash + extension)

    def ensure_persistence_location_exists(self) -> None:
        """
        Ensure that the location on the filesystem we will be writing to actually exists. If not, create it.
        """
        log_directory = self.log_directory
        # Create the folder if it doesn't exist
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

    def does_object_exist(self, path: str) -> bool:
        """
        Check to see if a path exists on the filesystem.
        :param path:
        :return:
        """
        return os.path.exists(path)

    def _get_log_directory(self) -> str:
        """
        Find a location on the filesystem to write our logs to.
        :return:
        """
        filename = "functions"

        # If explicitly defined
        env_dir = os.getenv(ENVVAR)
        if env_dir and os.path.isdir(env_dir):
            return os.path.join(env_dir, filename)

        # If installed as a library
        library_dir = os.path.join(user_data_dir(LIB_NAME), filename)
        if os.path.isdir(library_dir) or not os.path.exists(library_dir):
            return library_dir

        # If installed in a project that contains a git repo - place it in the same folder as the git repo
        current_dir = os.getcwd()
        while current_dir != os.path.root:
            if ".git" in os.listdir(current_dir):
                return os.path.join(current_dir, filename)
            current_dir = os.path.dirname(current_dir)

        return os.path.join(os.getcwd(), filename)

    def load_dataset(self, dataset_type, func_hash, return_type="both") -> Optional[int]:
        """
        Get the size of the dataset for a function hash
        """
        log_directory = self._get_log_directory()
        dataset_type_map = {"alignments": ALIGN_FILE_EXTENSION,
                            "positive": POSITIVE_FILE_EXTENSION,
                            "negative": NEGATIVE_FILE_EXTENSION,
                            "patches": PATCH_FILE_EXTENSION}

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

    def load_existing_datasets(self) -> Dict[str, Dict[str, str]]:
        log_directory = self.log_directory
        dataset_lengths = {
            SYMBOLIC_ALIGNMENTS: {},
            POSITIVE_EMBEDDABLE_ALIGNMENTS: {},
            NEGATIVE_EMBEDDABLE_ALIGNMENTS: {},
            PATCHES: {},
        }

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
            if ALIGN_FILE_EXTENSION not in file \
                    and PATCH_FILE_EXTENSION not in file \
                    and POSITIVE_FILE_EXTENSION not in file \
                    and NEGATIVE_FILE_EXTENSION not in file:
                continue
            elif ALIGN_FILE_EXTENSION in file:
                dataset_type = SYMBOLIC_ALIGNMENTS
            elif POSITIVE_FILE_EXTENSION in file:
                dataset_type = POSITIVE_EMBEDDABLE_ALIGNMENTS
            elif NEGATIVE_FILE_EXTENSION in file:
                dataset_type = NEGATIVE_EMBEDDABLE_ALIGNMENTS
            else:
                dataset_type = PATCHES
            func_hash = file.replace(ALIGN_FILE_EXTENSION, "").replace(PATCH_FILE_EXTENSION, "")
            dataset_lengths[dataset_type][func_hash] = -1
        return dataset_lengths

    def write(self, path: str, data: str, mode: Literal["w", "a", "a+b"] = "w") -> None:
        """
        Write data to a file
        """
        with open(path, mode) as f:
            f.write(data)

    def read(self, path: str) -> str:
        """
        Read data from a file
        """
        with open(path, "r") as f:
            return f.read()

    def get_hash_from_path(self, path) -> str:
        """
        Given a path with a hash, return only the hash
        :param path: The path to the file
        :return: The hash
        """
        return path.replace(PATCH_FILE_EXTENSION, ""). \
            replace(self.log_directory, ""). \
            lstrip("/"). \
            lstrip("\\")
