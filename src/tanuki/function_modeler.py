import ast
import datetime
import io
import json
from typing import List, Tuple, Dict

import openai

from tanuki.constants import EXAMPLE_ELEMENT_LIMIT, PATCHES, SYMBOLIC_ALIGNMENTS, POSITIVE_EMBEDDABLE_ALIGNMENTS, \
    NEGATIVE_EMBEDDABLE_ALIGNMENTS
from tanuki.models.function_description import FunctionDescription
from tanuki.models.function_example import FunctionExample
from tanuki.trackers.dataset_worker import DatasetWorker
from tanuki.utils import approximate_token_count, prepare_object_for_saving, encode_int, decode_int
import copy


class FunctionModeler(object):
    """
    This class manages the registered function models and their datasets
    comprised of symbolic and embeddable alignments, and symbolic and embeddable patches
    """

    def __init__(self, data_worker: DatasetWorker, environment_id=0) -> None:
        self.function_configs = {}
        self.data_worker = data_worker
        self.distillation_token_limit = 3000  # the token limit for finetuning
        self.symbolic_align_buffer = {}
        self.embeddable_align_buffer = {}
        self._get_datasets()
        self.environment_id = environment_id
        self.check_finetune_blacklist = []
        self.execute_finetune_blacklist = []
        self.store_data_blacklist = []

    def _get_dataset_info(self, dataset_type, func_hash, type="length"):
        """
        Get the dataset size for a function hash
        """
        return self.data_worker.load_dataset(dataset_type, func_hash, return_type=type)

    def _get_datasets(self):
        """
        Get the existing datasets from the data worker
        """
        self.dataset_sizes = self.data_worker.load_existing_datasets()

    def save_embeddable_align_statements(self,
                                         function_hash: str,
                                         args,
                                         kwargs,
                                         positive_pairs: List[Tuple[List, Dict]],
                                         negative_pairs: List[Tuple[List, Dict]]):
        """
        Save the contrastive align statements for the embeddable function.
        Do not save if the function hash is in the store data blacklist

        Args:
            function_hash: A unique hash for the function
            args: The arguments of the function
            kwargs: The keyword arguments of the function
            positive_pairs: A list of the other function invocations that are should have equivalent embeddings
            negative_pairs: A list of the other function invocations that are should have different embeddings
        """
        # prepare args and kwargs for saving
        copy_args = copy.deepcopy(args)
        copy_kwargs = copy.deepcopy(kwargs)
        parsed_args = prepare_object_for_saving(copy_args)
        parsed_kwargs = prepare_object_for_saving(copy_kwargs)

        # prepare positive pairs for saving
        parsed_positive_pairs = []
        for pair in positive_pairs:
            copy_pair = copy.deepcopy(pair)
            parsed_pair = prepare_object_for_saving(copy_pair)
            parsed_positive_pairs.append(parsed_pair)

        # prepare negative pairs for saving
        parsed_negative_pairs = []
        for pair in negative_pairs:
            copy_pair = copy.deepcopy(pair)
            parsed_pair = prepare_object_for_saving(copy_pair)
            parsed_negative_pairs.append(parsed_pair)

        # save the contrastive pairs
        for pair in parsed_positive_pairs:
            self._save_contrastive_alignment_pair(function_hash, parsed_args, parsed_kwargs, pair, positive=True)
        for pair in parsed_negative_pairs:
            self._save_contrastive_alignment_pair(function_hash, parsed_args, parsed_kwargs, pair, positive=False)

    def _save_contrastive_alignment_pair(self, function_hash: str, args, kwargs, pair, positive=True):
        """
        Save a contrastive pair
        """
        example = FunctionExample(args, kwargs, pair)
        if function_hash not in self.store_data_blacklist:
            successfully_saved, new_datapoint = self.data_worker.log_embeddable_align(function_hash, example, positive)
        else:
            successfully_saved = False
            new_datapoint = True
        if successfully_saved:
            if positive:
                if function_hash in self.dataset_sizes[POSITIVE_EMBEDDABLE_ALIGNMENTS]:
                    self.dataset_sizes[POSITIVE_EMBEDDABLE_ALIGNMENTS][function_hash] += 1
                else:
                    self.dataset_sizes[POSITIVE_EMBEDDABLE_ALIGNMENTS][function_hash] = 1
            if not positive:
                if function_hash in self.dataset_sizes[NEGATIVE_EMBEDDABLE_ALIGNMENTS]:
                    self.dataset_sizes[NEGATIVE_EMBEDDABLE_ALIGNMENTS][function_hash] += 1
                else:
                    self.dataset_sizes[NEGATIVE_EMBEDDABLE_ALIGNMENTS][function_hash] = 1

        if new_datapoint:
            # update align buffer
            if function_hash not in self.embeddable_align_buffer:
                self.embeddable_align_buffer[function_hash] = bytearray()
            self.embeddable_align_buffer[function_hash].extend(str(example.__dict__).encode('utf-8') + b'\r\n')

    def save_symbolic_align_statements(self, function_hash, args, kwargs, output):
        """
        Save the align statements and add to the align buffer
        Do not save if the function hash is in the store data blacklist
        Then just add the datapoints to the align buffer
        """
        # prepare output for saving and later parsing
        # make a deepcopy of the output to avoid changing the original object
        copy_output = copy.deepcopy(output)
        parsed_output = prepare_object_for_saving(copy_output)

        # prepare args and kwargs for saving
        copy_args = copy.deepcopy(args)
        copy_kwargs = copy.deepcopy(kwargs)
        parsed_args = prepare_object_for_saving(copy_args)
        parsed_kwargs = prepare_object_for_saving(copy_kwargs)

        example = FunctionExample(parsed_args, parsed_kwargs, parsed_output)
        if function_hash not in self.store_data_blacklist:
            successfully_saved, new_datapoint = self.data_worker.log_symbolic_align(function_hash, example)
        else:
            successfully_saved = False
            new_datapoint = True
        if successfully_saved:
            if function_hash in self.dataset_sizes[SYMBOLIC_ALIGNMENTS]:
                self.dataset_sizes[SYMBOLIC_ALIGNMENTS][function_hash] += 1
            else:
                self.dataset_sizes[SYMBOLIC_ALIGNMENTS][function_hash] = 1

        if new_datapoint:
            # update align buffer
            if function_hash not in self.symbolic_align_buffer:
                self.symbolic_align_buffer[function_hash] = bytearray()
            self.symbolic_align_buffer[function_hash].extend(str(example.__dict__).encode('utf-8') + b'\r\n')

    def save_symbolic_datapoint(self, func_hash, example):
        """
        Save datapoint to the training data
        """
        written_datapoints = self.data_worker.log_symbolic_patch(func_hash, example)
        for func_hash, datapoints in written_datapoints.items():
            if func_hash in self.dataset_sizes[PATCHES]:
                # if the dataset size is -1, it means we havent read in the dataset size yet
                if self.dataset_sizes[PATCHES][func_hash] == -1:
                    self.dataset_sizes[PATCHES][func_hash] = self._get_dataset_info(PATCHES, func_hash, type="length")
                else:
                    self.dataset_sizes[PATCHES][func_hash] += datapoints
            else:
                self.dataset_sizes[PATCHES][func_hash] = datapoints
        return len(written_datapoints) > 0

    def get_symbolic_alignments(self, func_hash, max=20):
        """
        Get all symbolic aligns for a function hash
        """

        if func_hash not in self.symbolic_align_buffer:
            return []

        buffer = self.symbolic_align_buffer[func_hash]
        return self._get_examples_from_alignment_buffer(buffer, max)

    def get_embeddable_alignments(self, func_hash, max=20):
        """
        Get all embeddable aligns for a function hash
        """

        if func_hash not in self.embeddable_align_buffer:
            return []

        buffer = self.embeddable_align_buffer[func_hash]
        return self._get_examples_from_alignment_buffer(buffer, max)

    def _get_examples_from_alignment_buffer(self, buffer, max=20):
        """
        Get examples from a buffer
        """

        split_buffer = bytes(buffer).split(b"\n")

        # byte array of stringed python dicts into dict objects
        example_set = set()
        for example in split_buffer:
            if example == b"":
                continue
            example_set.add(example)

        # easy and straightforward way to get nr of words (not perfect but doesnt need to be)
        # Can do the proper way of tokenizing later, it might be slower and we dont need 100% accuracy
        example_element_limit = EXAMPLE_ELEMENT_LIMIT

        examples = []
        for example_bytes in split_buffer:
            if example_bytes in example_set:
                nr_of_elements = approximate_token_count(example_bytes)
                example_element_limit -= nr_of_elements
                if example_element_limit < 0:
                    break
                example = example_bytes.decode('utf-8')
                # json load the example
                try:
                    example = json.loads(example)
                except:
                    example = ast.literal_eval(example)
                examples.append(example)
                example_set.remove(example_bytes)

        return list(examples)[:max]

    def load_symbolic_align_statements(self, function_hash):
        """
        Load all align statements
        First check the data storage blacklist,
        if the func hash is in the blacklist, then set the dataset size to 0 and the align buffer to empty bytearray
        """
        if function_hash in self.store_data_blacklist:
            self.dataset_sizes[SYMBOLIC_ALIGNMENTS][function_hash] = 0
            self.symbolic_align_buffer[function_hash] = bytearray()

        elif function_hash not in self.symbolic_align_buffer:
            dataset_size, align_dataset = self._get_dataset_info(SYMBOLIC_ALIGNMENTS, function_hash, type="both")
            if align_dataset:
                self.symbolic_align_buffer[function_hash] = bytearray(align_dataset)
            self.dataset_sizes[SYMBOLIC_ALIGNMENTS][function_hash] = dataset_size

    def postprocess_symbolic_datapoint(self, func_hash, function_description, example, repaired=True):
        """
        Postprocess the datapoint
        First check if the datapoint should be added to the training data
        Add the datapoint if it should be added
        Then check if the function should be finetuned and execute finetuning if it should
        """
        try:
            if func_hash not in self.store_data_blacklist:
                added = self.save_symbolic_datapoint(func_hash, example)
                if added:
                    self._update_datapoint_config(repaired, func_hash)
        except Exception as e:
            print(e)
            print("Could not add datapoint to training data")
        if func_hash not in self.execute_finetune_blacklist:
            self.check_for_finetuning(function_description, func_hash)

    def load_function_config(self, func_hash, function_description):
        """
        Load the config file for a function hash
        """
        config, default = self.data_worker.load_function_config(func_hash)
        if default and func_hash not in self.check_finetune_blacklist:
            finetuned, finetune_config = self._check_for_finetunes(function_description)
            if finetuned:
                config = finetune_config
        self.function_configs[func_hash] = config
        return config

    def _check_for_finetunes(self, function_description: FunctionDescription) -> Tuple[bool, Dict]:
        # This here should be discussed, what's the bestd way to do it
        # hash the function_hash into 16 characters (to embed it into the name of OpenAI finetunes, for later retrieval)

        finetune_hash = function_description.__hash__(purpose="finetune") + encode_int(self.environment_id)
        # List 10 fine-tuning jobs
        finetunes = openai.FineTuningJob.list(limit=1000)
        # Check if the function_hash is in the fine-tuning jobs
        # the finetunes are in chronological order starting from newest
        # So this gets the latest finetune
        for finetune in finetunes["data"]:
            # check if the finetune hash is in the fine-tuned model name
            if finetune["status"] == "succeeded" and finetune_hash in finetune["fine_tuned_model"]:
                try:
                    config = self._construct_config_from_finetune(finetune_hash, finetune)
                    # save the config
                    self.data_worker.update_function_config(function_description.__hash__(), config)
                    return True, config
                except:
                    return False, {}

        return False, {}

    def _construct_config_from_finetune(self, finetune_hash, finetune):
        model = finetune["fine_tuned_model"]
        # get the ending location of finetune hash in the model name
        finetune_hash_end = model.find(finetune_hash) + len(finetune_hash)
        # get the next character after the finetune hash
        next_char = model[finetune_hash_end]
        # get the number of training runs
        nr_of_training_runs = decode_int(next_char) + 1
        nr_of_training_points = (2 ** (nr_of_training_runs - 1)) * 200
        config = {
            "distilled_model": model,
            "current_model_stats": {
                "trained_on_datapoints": nr_of_training_points,
                "running_faults": []},
            "last_training_run": {"trained_on_datapoints": nr_of_training_points},
            "current_training_run": {},
            "teacher_models": ["gpt-4", "gpt-4-32k"],  # currently supported teacher models
            "nr_of_training_runs": nr_of_training_runs}

        return config

    def get_models(self, function_description):
        """
        Return the current model from the config file
        """
        func_hash = function_description.__hash__()
        if func_hash in self.function_configs:
            func_config = self.function_configs[func_hash]
        else:
            func_config = self.load_function_config(func_hash, function_description)

        # for backwards compatibility
        if "distilled_model" not in func_config:
            if func_config["current_model"] in func_config["teacher_models"]:
                distilled_model = ""
            else:
                distilled_model = func_config["current_model"]
        else:
            distilled_model = func_config["distilled_model"]

        return distilled_model, func_config["teacher_models"]

    def _update_datapoint_config(self, repaired, func_hash):
        """
        Update the config to reflect the new datapoint in the training data
        First adds 1 to the current datapoints
        Then updates running faults depending if priority is True or not and takes last 100
        Then checks the revert condition, i.e if last 10 datapoints are 50% faulty
        Finally updates the config file 
        Args:
           priority (bool): whether the datapoint was fixed by the teacher model/should be added to the training data
        """
        try:
            if repaired:
                self.function_configs[func_hash]["current_model_stats"]["running_faults"].append(1)
            else:
                self.function_configs[func_hash]["current_model_stats"]["running_faults"].append(0)
            # take the last 100 datapoints
            self.function_configs[func_hash]["current_model_stats"]["running_faults"] = \
                self.function_configs[func_hash]["current_model_stats"]["running_faults"][-100:]

            # check if the last 10 datapoints are 50% faulty, this is the switch condition
            if sum(self.function_configs[func_hash]["current_model_stats"]["running_faults"][-10:]) / 10 > 0.5:
                self.function_configs[func_hash]["distilled_model"] = ""
                self.function_configs[func_hash]["current_model_stats"]["trained_on_datapoints"] = 0
                self.function_configs[func_hash]["current_model_stats"]["running_faults"] = []
            self._update_config_file(func_hash)

        except Exception as e:
            print(e)
            print("Could not update config file")
            pass

    def _update_config_file(self, func_hash):
        self.data_worker.update_function_config(func_hash, self.function_configs[func_hash])

    def check_for_finetuning(self, function_description, func_hash):
        """
        Check for finetuning status
        If already finetuning, check for finetuning status
        If not finetuning, check for finetuning condition and execute finetuning if condition is met
        """
        try:
            # check if already finetuning
            if "job_id" in self.function_configs[func_hash]["current_training_run"]:
                # check for job status
                self._check_finetuning_status(func_hash)
            else:
                # check for finetuning condition
                if self._check_finetuning_condition(func_hash):
                    self._execute_finetuning(function_description, func_hash)
        except Exception as e:
            print(e)
            print("Error checking for finetuning")

    def _check_finetuning_condition(self, func_hash):
        """
        Check if the finetuning condition is met
        Currently finetuning condition is dependent on the number of symbolic datapoints since last finetuning
        """
        if func_hash not in self.function_configs:
            return False

        training_threshold = (2 ** self.function_configs[func_hash]["nr_of_training_runs"]) * 200

        align_dataset_size = self.dataset_sizes[SYMBOLIC_ALIGNMENTS][func_hash] if func_hash in self.dataset_sizes[
            SYMBOLIC_ALIGNMENTS] else 0
        patch_dataset_size = self.dataset_sizes[PATCHES][func_hash] if func_hash in self.dataset_sizes[PATCHES] else 0

        if patch_dataset_size == -1:
            # if havent read in the patch dataset size, read it in
            patch_dataset_size = self._get_dataset_info(PATCHES, func_hash, type="length")
            self.dataset_sizes[PATCHES][func_hash] = patch_dataset_size

        return (patch_dataset_size + align_dataset_size) > training_threshold

    def _execute_finetuning(self, function_description, func_hash):
        """
        Execute the finetuning
        First create the OpenAI compatible dataset with jsonL file and upload it
        Then submit the OpenAI finetuning job
        Finally update the config file to reflect the new finetuning job as current
        """
        # get function description
        function_string = str(function_description.__dict__.__repr__() + "\n")

        # get the align dataset
        align_dataset = self._get_dataset_info(SYMBOLIC_ALIGNMENTS, func_hash, type="dataset")
        if not align_dataset:
            align_dataset = ""
        else:
            align_dataset = align_dataset.decode('utf-8')

        # get the patch dataset
        patch_dataset = self._get_dataset_info(PATCHES, func_hash, type="dataset")
        if not patch_dataset:
            patch_dataset = ""
        else:
            patch_dataset = patch_dataset.decode('utf-8')

        if align_dataset == "" and patch_dataset == "":
            return

        dataset = align_dataset + patch_dataset

        dataset.replace("\\n", "[SEP_TOKEN]")
        dataset = dataset.split("\n")
        dataset = [x.replace("[SEP_TOKEN]", "\\n") for x in dataset if x != ""]
        # read in the dataset file
        dataset = [ast.literal_eval(x) for x in dataset]
        #
        # create the openai dataset
        instruction = "You are given below a function description and input data. The function description of what the function must carry out can be found in the Function section, with input and output type hints. The input data can be found in Input section. Using the function description, apply the function to the Input and return a valid output type, that is acceptable by the output_class_definition and output_class_hint. Return None if you can't apply the function to the input or if the output is optional and the correct output is None.\nINCREDIBLY IMPORTANT: Only output a JSON-compatible string in the correct response format."
        finetuning_dataset = [{"messages": [
            {
                "role": "system",
                "content": f"You are a skillful and accurate language model, who applies a described function on input data. Make sure the function is applied accurately and correctly and the outputs follow the output type hints and are valid outputs given the output types."
            },
            {"role": "user",
             "content": f"{instruction}\nFunction: {function_string}---\nInputs:\nArgs: {x['args']}\nKwargs: {x['kwargs']}\nOutput:"},
            {"role": "assistant", "content": str(x['output']) if x['output'] is not None else "None"}]}
            for x in dataset]

        # Create an in-memory text stream
        temp_file = io.StringIO()
        # Write data to the stream
        for idx, item in enumerate(finetuning_dataset):
            temp_file.write(json.dumps(item))
            if idx != len(finetuning_dataset) - 1:
                temp_file.write("\n")

        # Reset the stream position to the beginning
        temp_file.seek(0)

        # create the finetune hash
        finetune_hash = function_description.__hash__(purpose="finetune")
        nr_of_training_runs = self.function_configs[func_hash]["nr_of_training_runs"]
        finetune_hash += encode_int(self.environment_id)
        finetune_hash += encode_int(nr_of_training_runs)

        # Use the stream as a file
        try:
            response = openai.File.create(file=temp_file, purpose='fine-tune')
        except Exception as e:
            return

        # here can be sure that datasets were read in as that is checked in the finetune_check
        align_dataset_size = self.dataset_sizes[SYMBOLIC_ALIGNMENTS][func_hash] if func_hash in self.dataset_sizes[
            SYMBOLIC_ALIGNMENTS] else 0
        patch_dataset_size = self.dataset_sizes[PATCHES][func_hash] if func_hash in self.dataset_sizes[PATCHES] else 0
        total_dataset_size = align_dataset_size + patch_dataset_size
        training_file_id = response["id"]
        # submit the finetuning job
        try:
            finetuning_response = openai.FineTuningJob.create(training_file=training_file_id, model="gpt-3.5-turbo",
                                                              suffix=finetune_hash)
        except Exception as e:
            return

        self.function_configs[func_hash]["current_training_run"] = {"job_id": finetuning_response["id"],
                                                                    "trained_on_datapoints": total_dataset_size,
                                                                    "last_checked": datetime.datetime.now().strftime(
                                                                        "%Y-%m-%d %H:%M:%S")}
        # update the config json file
        try:
            self._update_config_file(func_hash)
        except Exception as e:
            print(e)
            print("Could not update config file to register a finetuning run")

    def _check_finetuning_status(self, func_hash):
        """
        Check the status of the current finetuning job
        If the job is finished, update the config file to reflect the new model
        """

        job_id = self.function_configs[func_hash]["current_training_run"]["job_id"]
        last_checked = self.function_configs[func_hash]["current_training_run"]["last_checked"]
        # check if last checked was more than 30 mins ago
        if (datetime.datetime.now() - datetime.datetime.strptime(last_checked,
                                                                 "%Y-%m-%d %H:%M:%S")).total_seconds() > 1800:
            response = openai.FineTuningJob.retrieve(job_id)
            self.function_configs[func_hash]["current_training_run"]["last_checked"] = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S")
            if response["status"] == "succeeded" or response["status"] == "failed":
                self._update_finetune_config(response, func_hash, response["status"])
            else:
                self._update_config_file(func_hash)

    def _update_finetune_config(self, response, func_hash, status):
        """
        Update the config file to reflect the new model and switch the current model to the finetuned model
        """
        if status == "failed":
            self.function_configs[func_hash]["current_training_run"] = {}
        else:
            self.function_configs[func_hash]["distilled_model"] = response["fine_tuned_model"]
            self.function_configs[func_hash]["last_training_run"] = self.function_configs[func_hash][
                "current_training_run"]
            self.function_configs[func_hash]["current_model_stats"] = {
                "trained_on_datapoints": self.function_configs[func_hash]["current_training_run"][
                    "trained_on_datapoints"],
                "running_faults": []}
            self.function_configs[func_hash]["nr_of_training_runs"] += 1
            self.function_configs[func_hash]["current_training_run"] = {}
        try:
            self._update_config_file(func_hash)
        except Exception as e:
            print(e)
            print("Could not update config file after a successful finetuning run")
            pass
