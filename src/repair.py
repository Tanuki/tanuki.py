
import openai
import json


def repair_output(args, kwargs, function_description, choice, validator, function_modeler, language_modeler):
        teacher_models = function_modeler.get_models(function_description.__hash__())[1]
        valid = False
        retry_index = 5
        f = str(function_description.__dict__.__repr__() + "\n")
        error = f"Output type was not valid. Expected an valid object of type {function_description.output_type_hint}, got '{choice}'"
        failed_outputs_list = [(choice, error)]
        while retry_index > 0 and not valid:
            
            aligns = function_modeler.get_alignments(function_description.__hash__(), max=5)
            examples = "\n".join([f"Inputs:\nArgs: {align['args']}\nKwargs {align['kwargs']}\nOutput: {align['output']}" for align in aligns])
            choice = language_modeler.repair_generate(args, kwargs, f, failed_outputs_list, examples, teacher_models)
            if not choice:
                # if no choice then the input was too long for the model
                # no specific error but the retry index goes down
                retry_index -= 1
                continue

            # start parsing the object, WILL NEED TO BE CHANGED, VERY HACKY
            try:
                # json load
                choice_parsed = json.loads(choice)
            except:
                # if it fails, it's not a json object, try eval
                try:
                    choice_parsed = eval(choice)
                except: 
                    choice_parsed = choice

            valid = validator.check_type(choice_parsed, function_description.output_type_hint)
            if not valid:
                error = f"Output type was not valid. Expected an object of type {function_description.output_type_hint}, got '{choice}'"
                failed_outputs_list.append((choice, error))
                retry_index -= 1

        return choice, choice_parsed, valid