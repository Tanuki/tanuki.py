import os

from monkey import ALIGN_FILE_NAME
from trackers.base_logger import BaseLogger


class TextFileLogger(BaseLogger):

    def _log_align(self, message, example):
        log_directory = os.path.join(os.getcwd(), ALIGN_FILE_NAME)
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        log_file_path = os.path.join(log_directory, message)
        with open(log_file_path, "a") as f:
            f.write(str(example.__dict__) + "\n")

    def _log_patch(self, message, example):
        log_directory = os.path.join(os.getcwd(), ALIGN_FILE_NAME)
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        log_file_path = os.path.join(log_directory, message)
        with open(log_file_path, "a") as f:
            f.write(str(example.__dict__) + "\n")