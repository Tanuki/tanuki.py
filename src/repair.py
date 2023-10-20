
import openai
import json


def repair_output(input, function_description, choice, error, validator, example_input):

        valid = False
        retry_index = 5
        f = str(function_description.__dict__.__repr__() + "\n")
        instruction = "Below are an outputs of a function applied to inputs, which failed type validation. The input to the function is brought out in the INPUT section and function description is brought out in the FUNCTION DESCRIPTION section. Your task is to apply the function to the input and return a correct output in the right type. The FAILED EXAMPLES section will show previous outputs of this function applied to the data, which failed type validation and hence are wrong outputs. Using the input and function description output the accurate output following the output_class_definition and output_type_hint attributes of the function description, which define the output type. Make sure the output is an accurate function output and in the correct type. Return None if you can't apply the function to the input or if the output is optional and the correct output is None."
        failed_outputs_list = [(choice, error)]
        while retry_index > 0 and not valid:
            
            failed_examples = ""
            for failed_output in failed_outputs_list:
                failed_examples += f"Output: {failed_output[0]}\nError: {failed_output[1]}\n\n"
            content = f"{instruction}\nFUNCTION DESCRIPTION: {f}\n{example_input}---INPUT: {input}\nFAILED EXAMPLES: {failed_examples}Correct output:"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=0,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            choice = response.choices[0].message.content.strip("'")

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