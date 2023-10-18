import json
import os
import openai


class Modeler():
    def __init__(self, function_name, config_location = "function_configs"):
        """
        Initialise the modeler
        Args:
            function_name (str): name of the function, used to read the config file from the config_location
            config_location (str): location of the config files, default is function_configs
        
        """
        self.function_name = function_name
        self.config_location = config_location
        self._get_config()
        self.teacher_models = ["gpt-4"]

    def _get_config(self):
        """
        Get the config file for the function. Uses the function_name and config_location to find the config file
        Config file has to have the name function_config.json 
        """
        config_path = os.path.join(self.config_location, self.function_name, "function_config.json")
        with open(config_path) as f:
            self.config = json.load(f)
    
    def get_model(self):
        """
        Return the current model from the config file
        """
        return self.config["current_model"]
    
    def add_datapoint_to_training_data(self, datapoint):
        """
        Adds datapoint to the training data
        Only adds datapoint if the current model is a teacher model or if the datapoint is marked as priority
        Priority means it was fixed by the teacher model itself
        
        Args:
            datapoint (dict): datapoint to add to the training data
            Keys:
                content (str): the input content of the datapoint
                choice (str): the output of the datapoint
                priority (bool): whether the datapoint was fixed by the teacher model/should be added to the training data
        """
        path = os.path.join(self.config_location, self.function_name, "training_data.jsonl")
        if self.config["current_model"] in self.teacher_models or datapoint["priority"]:
            with open(path, "a") as f:
                f.write(json.dumps(datapoint) + "\n")
            self._update_datapoint_config(datapoint["priority"])
        

    
    def postprocess_datapoint(self, datapoint):
        """
        Postprocess the datapoint
        First check if datapoint should be added to training data
        Then check for finetuning conditions
        Args:
            datapoint (dict): datapoint to add to the training data
            Keys:
                content (str): the input content of the datapoint
                choice (str): the output of the datapoint
                priority (bool): whether the datapoint was fixed by the teacher model/should be added to the training data
        """
        try:
            self.add_datapoint_to_training_data(datapoint)
        except Exception as e:
            print(e)
            print("Could not add datapoint to training data")
            return None
        
        self.check_for_finetuning()


    def check_for_finetuning(self):
        """
        Check for finetuning status
        If already finetuning, check for finetuning status
        If not finetuning, check for finetuning condition and execute finetuning if condition is met
        """
        try:
            # check if already finetuning
            if "job_id" in self.config["current_training_run"]:
                # check for job status
                self._check_finetuning_status()
            else:
            # check for finetuning condition
                if self._check_finetuning_condition():
                    self._execute_finetuning()
        except Exception as e:
            print(e)
            print("Error checking for finetuning")
    
    def _check_finetuning_condition(self):
        """
        Check if the finetuning condition is met
        Currently finetuning condition is dependent on the number of datapoints since last finetuning
        """
        last_training_run_datapoints = self.config["last_training_run"]["trained_on_datapoints"]
        return self.config["current_datapoints"] - last_training_run_datapoints > 10*(2**self.config["nr_of_training_runs"])
    
    def _execute_finetuning(self):
        """
        Execute the finetuning
        First create the OpenAI compatible dataset with jsonL file and upload it
        Then submit the OpenAI finetuning job
        Finally update the config file to reflect the new finetuning job as current
        """
        training_dataset = os.path.join(self.config_location, self.function_name, "training_data.jsonl")
        # read in the dataset file
        dataset = open(training_dataset, 'r', encoding='utf-8').readlines()
        dataset = [json.loads(x) for x in dataset]
        #
        # create the openai dataset
        finetuning_dataset = [{"messages":[
                        {
                            "role": "system",
                            "content": "You are a skillful assistant who carries out the user instructions in a correct and accurate manner",
                        },
                        {"role": "user", "content": x["content"]},
                        {"role": "assistant", "content": x["choice"]}]}
                        for x in dataset]
        
        # create the openai file
        temp_file_path = os.path.join(self.config_location, self.function_name, "finetuning_data_temp.jsonl")
        with open(temp_file_path, "w", encoding='utf-8') as f:
            for idx, item in enumerate(finetuning_dataset):
                f.write(json.dumps(item))
                if idx != len(finetuning_dataset) - 1:
                    f.write("\n")

        response = openai.File.create(file=open(temp_file_path, 'r', encoding='utf-8'), purpose='fine-tune')
        training_file_id = response["id"]
        finetuning_response = openai.FineTuningJob.create(training_file=training_file_id, model="gpt-3.5-turbo", suffix = "test_autom")
        # delete the temp file
        os.remove(temp_file_path)
        
        self.config["current_training_run"] = {"job_id": finetuning_response["id"], "trained_on_datapoints": self.config["current_datapoints"]}
        # update the config json file
        try:
            self._update_config_file()
        except Exception as e:
            print(e)
            print("Could not update config file to register a finetuning run")
        
        #try:
        #    # update the training data jsonL such that all datapoints are marked as not priority
        #    with open(training_dataset, "r") as f:
        #        training_data = [json.loads(x) for x in f.readlines()]
        #    for idx, datapoint in enumerate(training_data):
        #        training_data[idx]["priority"] = False
        #    with open(training_dataset, "w") as f:
        #        for idx, datapoint in enumerate(training_data):
        #            f.write(json.dumps(datapoint))
        #            if idx != len(training_data) - 1:
        #                f.write("\n")
        #except Exception as e:
        #    print(e)
        #    print("Could not update training data file to to remove priorities")


    def _check_finetuning_status(self):
        """
        Check the status of the current finetuning job
        If the job is finished, update the config file to reflect the new model
        """
        job_id = self.config["current_training_run"]["job_id"]
        response = openai.FineTuningJob.retrieve(job_id)
        if response["status"] == "succeeded":
            self._update_finetune_config(response)
    
    def _update_finetune_config(self, response):
        """
        Update the config file to reflect the new model and switch the current model to the finetuned model
        """
        self.config["current_model"] = response["fine_tuned_model"]
        self.config["last_training_run"] = self.config["current_training_run"]
        self.config["current_model_stats"] = {"trained_on_datapoints": self.config["current_training_run"]["trained_on_datapoints"], "running_faults": []}
        self.config["nr_of_training_runs"] += 1
        self.config["current_training_run"] = {}
        try:
            self._update_config_file()
        except Exception as e:
            print(e)
            print("Could not update config file after a successful finetuning run")
            pass

    def _update_datapoint_config(self, priority):
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
            self.config["current_datapoints"] += 1
            if priority:
                self.config["current_model_stats"]["running_faults"].append(1)
            else:
                self.config["current_model_stats"]["running_faults"].append(0)
            # take the last 100 datapoints
            self.config["current_model_stats"]["running_faults"] = self.config["current_model_stats"]["running_faults"][-100:]

            # check if the last 10 datapoints are 50% faulty, this is the switch condition
            if sum(self.config["current_model_stats"]["running_faults"][-10:])/10 > 0.5:
                self.config["current_model"] = self.teacher_models[0]
                self.config["current_model_stats"]["trained_on_datapoints"] = 0
                self.config["current_model_stats"]["running_faults"] = []

            self._update_config_file()
        except Exception as e:
            print(e)
            print("Could not update config file")
            pass

    def _update_config_file(self):
        """
        Update the config file with the new config
        """
        with open(os.path.join(self.config_location, self.function_name, "function_config.json"), "w") as f:
                json.dump(self.config, f)



## TESTING
if __name__ == "__main__":
    pass