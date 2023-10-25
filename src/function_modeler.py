from utils import approximate_token_count
import openai
import datetime
from models.language_model_output import LanguageModelOutput
import os
import ast
import json
import io
from models.function_example import FunctionExample

EXAMPLE_ELEMENT_LIMIT = 1000

class FunctionModeler(object):
    def __init__(self, data_worker) -> None:
        self.function_configs = {}
        self.data_worker = data_worker
        self.distillation_token_limit = 3000 # the token limit for finetuning
        self.align_buffer = {}
        self._get_dataset_sizes()
    

    def _get_dataset_sizes(self):
        """
        Get the dataset sizes from the data worker
        """
        self.dataset_sizes = self.data_worker._load_dataset_sizes()

    def save_align_statements(self, function_hash, args, kwargs, output):
        """
        Save the align statements and add to the align buffer
        """
        example = FunctionExample(args, kwargs, output)

        self.data_worker.log_align(function_hash, example)
        if function_hash in self.dataset_sizes["alignments"]:
            self.dataset_sizes["alignments"][function_hash] += 1 
        else:
            self.dataset_sizes["alignments"][function_hash] = 1
        
        # update align buffer
        if function_hash not in self.align_buffer:
            self.align_buffer[function_hash] = bytearray()
        self.align_buffer[function_hash].extend(str(example.__dict__).encode('utf-8') + b'\r\n')

    
    def save_datapoint(self, func_hash, example):
        """
        Save datapoint to the training data
        """
        written_datapoints = self.data_worker.log_patch(func_hash, example)
        for func_hash, datapoints in written_datapoints.items():
            if func_hash in self.dataset_sizes["patches"]:
                self.dataset_sizes["patches"][func_hash] += datapoints
            else:
                self.dataset_sizes["patches"][func_hash] = datapoints
        return len(written_datapoints) > 0
    
    def get_alignments(self, func_hash, max=20):
        """
        Get all aligns for a function hash
        """

        if func_hash not in self.align_buffer:
            return []

        buffer = self.align_buffer[func_hash]

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
                example = ast.literal_eval(example)
                examples.append(example)
                example_set.remove(example_bytes)

        return list(examples)[:max]

    def load_align_statements(self):
        """
        Load all align statements
        """
        self.align_buffer = self.data_worker.load_alignments()


    def postprocess_datapoint(self, func_hash, function_description, example, repaired=True):
        """
        Postprocess the datapoint
        """
        try:
            
            added = self.save_datapoint(func_hash, example)
            if added:
                self._update_datapoint_config(repaired, func_hash)
        except Exception as e:
            print(e)
            print("Could not add datapoint to training data")
            return None

        self.check_for_finetuning(function_description, func_hash)

    def _load_function_config(self, func_hash):
        """
        Load the config file for a function hash
        """
        
        config = self.data_worker._load_function_config(func_hash)
        self.function_configs[func_hash] = config
        return config
            
    
    def get_models(self, func_hash):
        """
        Return the current model from the config file
        """
    
        if func_hash in self.function_configs:
            func_config = self.function_configs[func_hash]
        else:
            func_config = self._load_function_config(func_hash)
        
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
        self.data_worker._update_function_config(func_hash, self.function_configs[func_hash])


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
        Currently finetuning condition is dependent on the number of datapoints since last finetuning
        """
        if func_hash not in self.function_configs:
            return False

        last_training_run_datapoints = self.function_configs[func_hash]["last_training_run"]["trained_on_datapoints"]

        training_threshold = (2 ** self.function_configs[func_hash]["nr_of_training_runs"]) * 200

        align_dataset_size = self.dataset_sizes["alignments"][func_hash] if func_hash in self.dataset_sizes["alignments"] else 0
        patch_dataset_size = self.dataset_sizes["patches"][func_hash] if func_hash in self.dataset_sizes["patches"] else 0

        return (patch_dataset_size + align_dataset_size) - last_training_run_datapoints > training_threshold
    
    def _execute_finetuning(self, function_description, func_hash):
        """
        Execute the finetuning
        First create the OpenAI compatible dataset with jsonL file and upload it
        Then submit the OpenAI finetuning job
        Finally update the config file to reflect the new finetuning job as current
        """

        align_dataset, patch_dataset = self.data_worker.load_datasets(func_hash)
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
             "content": f"{instruction}\nFunction: {function_description}---\nInputs:\nArgs: {x['args']}\nKwargs: {x['kwargs']}\nOutput:"},
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

        # Use the stream as a file
        response = openai.File.create(file=temp_file, purpose='fine-tune')

        align_dataset_size = self.dataset_sizes["alignments"][func_hash] if func_hash in self.dataset_sizes["alignments"] else 0
        patch_dataset_size = self.dataset_sizes["patches"][func_hash] if func_hash in self.dataset_sizes["patches"] else 0
        total_dataset_size = align_dataset_size + patch_dataset_size

        training_file_id = response["id"]
        finetuning_response = openai.FineTuningJob.create(training_file=training_file_id, model="gpt-3.5-turbo",
                                                          suffix="test_autom")
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
            self.function_configs[func_hash]["last_training_run"] = self.function_configs[func_hash]["current_training_run"]
            self.function_configs[func_hash]["current_model_stats"] = {
                "trained_on_datapoints": self.function_configs[func_hash]["current_training_run"]["trained_on_datapoints"],
                "running_faults": []}
            self.function_configs[func_hash]["nr_of_training_runs"] += 1
            self.function_configs[func_hash]["current_training_run"] = {}
        try:
            self._update_config_file(func_hash)
        except Exception as e:
            print(e)
            print("Could not update config file after a successful finetuning run")
            pass