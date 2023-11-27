from abc import abstractmethod
from logging import Logger

from tanuki.models.function_example import FunctionExample


class DatasetWorker(Logger):
    def __init__(self, name, level=15):
        super().__init__(name, level)


    @abstractmethod
    def load_existing_datasets(self):
        """
        Get all existing datasets for functions
        Output must be a dictionary with the following structure:
        {
            "alignments": {
                "func_hash": -1
            },
            "patches": {
                "func_hash": -1
            }
        }
        Returns:
            dict: dictionary with the structure above 
        """
        pass

    @abstractmethod
    def log_symbolic_align(self, func_hash, *args, **kws):
        """
        Log an alignment statement to the dataset defined by func_hash
        Args:
            func_hash (str): the function hash
            *args: the args for the datapoint, where args[0] is a FunctionExample(args, kwargs, output) object
            **kws: the kwargs for the datapoint
        """
        pass

    @abstractmethod
    def log_embeddable_align(self, func_hash, first_example: FunctionExample, second_example: FunctionExample, **kws):
        """
        Log a contrastive statement to the dataset defined by func_hash
        Args:
            second_example:
            first_example:
            func_hash (str): the function hash
            *args: the args for the datapoint, where args[0] is a FunctionExample(args, kwargs, output) object
            **kws: the kwargs for the datapoint
        """
        pass

    @abstractmethod
    def log_symbolic_patch(self, func_hash, example):
        """
        Save the example to the patch dataset for the function hash
        Output must be a dictionary with the following structure:
        {
            "func_hash": int
        }
        Where func_hash is the hash of the function and int is the number of datapoints written to the dataset for this function
        
        Args:
            func_hash (str): the function hash
            example (FunctionExample): the example to be saved
        
        Returns:
            dict: dictionary with the structure above
        
        """

    @abstractmethod
    def load_function_config(self, func_hash):

        """
        Get the config file for the function.
        Function config must be a dictionary and have the following structure:
            distilled_model (str): distilled_model_name ("" if no distilled model),
            current_model_stats (dict): dict for current model stats
                example:
                {
                    "trained_on_datapoints" (int): 12 (number of datapoints trained on, 0 if not trained yet),
                    "running_faults" (list): [0, 0, 1] (list of 0s and 1s, where 0 is no fault and 1 is fault)
                }
            
            last_training_run (dict): dict for the last training run
                example:
                {
                    "job_id" (str): job_id for last training run,
                    "trained_on_datapoints" (int): dataset_size that was trained on,
                    "last_checked" (datetime in "%Y-%m-%d %H:%M:%S"): When the last check was made for status of training run)
                }
                Example when no training has been done yet:
                {
                    "trained_on_datapoints": 0
                }

            current_training_run (dict): Same structure as last_training_run, only is non-empty if currently a model is training
                Example when no training has been done yet:
                {}

            teacher_models (list of string): list of teacher models
                example:
                ["gpt-4", "gpt-4-32k"]

            nr_of_training_runs (int): number of training runs that have been done in total
            }
        
        The config file must be returned as a dictionary

        Args:
            func_hash (str): the function hash
        Returns:
            dict: the function config
        """
        pass

    @abstractmethod
    def load_dataset(self, dataset_type, func_hash, return_type):
        """
        Load the datasets for a function hash
        The datasets loaded must be a string, where different datapoints are on new lines
        The output depends on the return type, is it either dataset, dataset_length or both
        
        Args:
            dataset_type (str): either "alignments" or "patches"
            func_hash (str): the function hash
            return_type (str): the return type, either "dataset", "length" or "both"
        Returns:
            tuple: tuple of the form (dataset, length) if return_type is "both", otherwise just dataset or length
        """
        pass

    @abstractmethod
    def update_function_config(self, func_hash, config_to_be_saved):
        """
        Save the config file using the function hash to data storage
        Args:
            func_hash (str): the function hash
            config_to_be_saved (dict): the config to be saved
        
        """
        pass
