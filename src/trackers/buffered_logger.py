import os
from logging import Logger
import json
from appdirs import user_data_dir

from bloom_filter import BloomFilter, optimal_bloom_filter_params
from models.function_example import FunctionExample
import openai
import ast
import io
import datetime
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
        self.flush_limit = {}
        self.log_directory = self._get_log_directory()
        self.bloom_filter = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))
        self.configs = {}
        try:
            self.bloom_filter.load(self.log_directory)
        except FileNotFoundError:
            self.bloom_filter = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))

        self.write_count = 0
        self.write_limit = 1000  # Save the Bloom filter every 1000 writes
        self._get_dataset_sizes()
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

    def _get_dataset_sizes(self):
        # get all the files in the log directory
        files = os.listdir(self.log_directory)
        # discard all .json files
        files = [x for x in files if ".json" not in x]
        self.dataset_lengths = {}
        for file in files:
            with open(os.path.join(self.log_directory, file), "rb") as f:
                dataset = f.read().decode('utf-8')
            # get the total nr of /n in the file
            log_file_path = os.path.join(self.log_directory, file)
            self.dataset_lengths[log_file_path] = dataset.count("\n") - dataset.count("\\n") 

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
            return False

        self.miss_count += 1
        # Add to Bloom Filter
        self.bloom_filter.add(example_data.decode('utf-8'))

        log_directory = self._get_log_directory()
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        log_file_path = os.path.join(log_directory, message)

        if log_file_path not in self.buffers:
            self.buffers[log_file_path] = bytearray()
            self.flush_limit[log_file_path] = 1

        self.buffers[log_file_path].extend(example_data)

        self.write_count += 1

        if self.write_count >= self.write_limit:
            self.flush()
            self.save_bloom_filter()
            self.write_count = 0  # Reset counter

        if len(self.buffers[log_file_path]) >= self.flush_limit[log_file_path]:  # Flush after reaching 4KB
            with open(log_file_path, "a+b") as f:
                f.write(self.buffers[log_file_path])
            if log_file_path in self.dataset_lengths:
                self.dataset_lengths[log_file_path] += 1
            else:
                self.dataset_lengths[log_file_path] = 1
            self.buffers[log_file_path].clear()
            self.flush_limit[log_file_path] = 2 * self.flush_limit[log_file_path]
        return True

    def save_bloom_filter(self):
        self.bloom_filter.save(self.log_directory)

    def flush(self):
        for log_file_path, buffer in self.buffers.items():
            if len(buffer) > 0:
                with open(log_file_path, "a+b") as f:
                    f.write(buffer)
                self.dataset_lengths[log_file_path] = len(buffer)
                buffer.clear()

    def get_configs(self, log_file_path):

        """
        Get the config file for the function. Uses the message and log directory
        Config file has to have to be in .json
        """

        
        config_path = f"{log_file_path}.json"
        if not os.path.exists(config_path):
            self.configs[log_file_path] = {"current_model": "gpt-4", 
                    "current_model_stats": {"trained_on_datapoints": 0,
                    "running_faults": []}, 
                    "last_training_run": {"trained_on_datapoints": 0}, 
                    "current_training_run": {}, 
                    "current_datapoints": 0,
                    "teacher_models" : ["gpt-4"],
                    "nr_of_training_runs": 0}
            with open(config_path, "w") as f:
                json.dump(self.configs[log_file_path], f)
        else:
            with open(config_path, "r") as f:
                self.configs[log_file_path] = json.load(f)
        


    def get_model(self, message):
        """
        Return the current model from the config file
        """
        log_directory = self._get_log_directory()
        log_file_path = os.path.join(log_directory, message)

        if not os.path.exists(log_directory):
            os.makedirs(log_directory)


        self.get_configs(log_file_path)

        return self.configs[log_file_path]["current_model"]
    
    def postprocess_datapoint(self, message, function_description, example, log = True):
        """
        Postprocess the datapoint
        First check if datapoint should be added to training data
        Then check for finetuning conditions
        Args:
            ???
        """

        try:
            log_directory = self._get_log_directory()
            log_file_path = os.path.join(log_directory, message)
            if log or self.configs[log_file_path]["current_model"] in self.configs[log_file_path]["teacher_models"]:
                added = self.log_patch(message, example)
                if added:
                    self._update_datapoint_config(log, log_file_path)
        except Exception as e:
            print(e)
            print("Could not add datapoint to training data")
            return None
        
        self.check_for_finetuning(function_description, log_file_path)
    
    def check_for_finetuning(self, function_description, log_file_path):
        """
        Check for finetuning status
        If already finetuning, check for finetuning status
        If not finetuning, check for finetuning condition and execute finetuning if condition is met
        """
        try:
            # check if already finetuning
            if "job_id" in self.configs[log_file_path]["current_training_run"]:
                # check for job status
                self._check_finetuning_status(log_file_path)
            else:
            # check for finetuning condition
                if self._check_finetuning_condition(log_file_path):
                    self._execute_finetuning(function_description, log_file_path)
        except Exception as e:
            print(e)
            print("Error checking for finetuning")

    def _check_finetuning_condition(self, log_file_path):
        """
        Check if the finetuning condition is met
        Currently finetuning condition is dependent on the number of datapoints since last finetuning
        """
        last_training_run_datapoints = self.configs[log_file_path]["last_training_run"]["trained_on_datapoints"]   
        return self.dataset_lengths[log_file_path] - last_training_run_datapoints > 100*(2**self.configs[log_file_path]["nr_of_training_runs"])

    def _execute_finetuning(self, function_description, log_file_path):
        """
        Execute the finetuning
        First create the OpenAI compatible dataset with jsonL file and upload it
        Then submit the OpenAI finetuning job
        Finally update the config file to reflect the new finetuning job as current
        """
        # read in the dataset file
        with open (log_file_path, "rb") as f:
            dataset = f.read().decode('utf-8')
        dataset.replace("\\n", "[SEP_TOKEN]")
        dataset = dataset.split("\n")
        dataset = [x.replace("[SEP_TOKEN]", "\\n") for x in dataset if x != ""]
        # read in the dataset file
        dataset = [ast.literal_eval(x) for x in dataset]
        #
        # create the openai dataset
        instruction = "Optionally convert the input into the output type, using the docstring as a guide. Return None if you can't."
        warning = "INCREDIBLY IMPORTANT: Only output a JSON-compatible string in the correct response format."
        finetuning_dataset = [{"messages":[
                        {
                            "role": "system",
                            "content": "You are a skillful assistant who carries out the user instructions in a correct and accurate manner",
                        },
                        {"role": "user", "content": f"{instruction}\n{warning}\nFunction: {function_description}\nInput: {x['args']}\nOutput:"},
                        {"role": "assistant", "content": x['output'] if x['output'] is not None else "None"}]}
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


        training_file_id = response["id"]
        finetuning_response = openai.FineTuningJob.create(training_file=training_file_id, model="gpt-3.5-turbo", suffix = "test_autom")
        self.configs[log_file_path]["current_training_run"] = {"job_id": finetuning_response["id"],
                                                "trained_on_datapoints": self.configs[log_file_path]["current_datapoints"],
                                                "last_checked": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        # update the config json file
        try:
            self._update_config_file(log_file_path)
        except Exception as e:
            print(e)
            print("Could not update config file to register a finetuning run")


    def _check_finetuning_status(self, log_file_path):
        """
        Check the status of the current finetuning job
        If the job is finished, update the config file to reflect the new model
        """

        job_id = self.configs[log_file_path]["current_training_run"]["job_id"]
        last_checked = self.configs[log_file_path]["current_training_run"]["last_checked"]
        # check if last checked was more than 30 mins ago
        if (datetime.datetime.now() - datetime.datetime.strptime(last_checked, "%Y-%m-%d %H:%M:%S")).total_seconds() > 1800:
            response = openai.FineTuningJob.retrieve(job_id)
            self.configs[log_file_path]["current_training_run"]["last_checked"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if response["status"] == "succeeded":
                self._update_finetune_config(response, log_file_path)
            else: 
                self._update_config_file(log_file_path)

    def _update_finetune_config(self, response, log_file_path):
        """
        Update the config file to reflect the new model and switch the current model to the finetuned model
        """
        self.configs[log_file_path]["current_model"] = response["fine_tuned_model"]
        self.configs[log_file_path]["last_training_run"] = self.configs[log_file_path]["current_training_run"]
        self.configs[log_file_path]["current_model_stats"] = {"trained_on_datapoints": self.configs[log_file_path]["current_training_run"]["trained_on_datapoints"], "running_faults": []}
        self.configs[log_file_path]["nr_of_training_runs"] += 1
        self.configs[log_file_path]["current_training_run"] = {}
        try:
            self._update_config_file(log_file_path)
        except Exception as e:
            print(e)
            print("Could not update config file after a successful finetuning run")
            pass

    def _update_datapoint_config(self, priority, log_file_path):
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
            if priority:
                self.configs[log_file_path]["current_model_stats"]["running_faults"].append(1)
            else:
                self.configs[log_file_path]["current_model_stats"]["running_faults"].append(0)
            # take the last 100 datapoints
            self.configs[log_file_path]["current_model_stats"]["running_faults"] = self.configs[log_file_path]["current_model_stats"]["running_faults"][-100:]

            # check if the last 10 datapoints are 50% faulty, this is the switch condition
            if sum(self.configs[log_file_path]["current_model_stats"]["running_faults"][-10:])/10 > 0.5:
                self.configs[log_file_path]["current_model"] = self.configs[log_file_path]["teacher_models"][0]
                self.configs[log_file_path]["current_model_stats"]["trained_on_datapoints"] = 0
                self.configs[log_file_path]["current_model_stats"]["running_faults"] = []
            self._update_config_file(log_file_path)

        except Exception as e:
            print(e)
            print("Could not update config file")
            pass

    def _update_config_file(self, log_file_path):
        """
        Update the config file with the new config
        """
        config_to_be_saved = self.configs[log_file_path]
        with open(f"{log_file_path}.json", "w") as f:
                json.dump(config_to_be_saved, f)