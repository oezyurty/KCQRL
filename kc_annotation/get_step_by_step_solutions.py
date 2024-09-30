import json
from openai import OpenAI
import time
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--original_question_file', type=str, help='Path to the original question file')
parser.add_argument('--annotated_question_file', type=str, help='Path to the annotated question file')
args = parser.parse_args()

original_question_file = args.original_question_file
annotated_question_file = args.annotated_question_file

client = OpenAI()

system_prompt = """Your task is to generate the clear and concise step by step solutions of the provided Math problem. Please consider the below instructions in your generation. 

- You will be provided with the final answer, and additional Chinese explanation of the solution. When generating the step by step solution, you can leverage those information pieces, but you can also use your own judgment.  
- It is important that your generated step by step solution should be understandable as stand-alone, meaning that the student should not need to additionally check final answer or explanation provided. 
- Please provide your step-by-step solution as each step in a new line. Don't enumerate the steps. Don't put any bullet point. Separate the solution steps only with one newline \n . 
- Don't generate any text other than the step by step solution described earlier.
- Make your step-by-step solution concise as described earlier."""

user_prompt_template="""Question: {}
Final Answer: {}
Explanation: {}
Step by Step Solution: """

def structure_answer(item):
    """Function for handling the answer structure of different types of questions.
    If the question is fill-in-the-blank, 填空, it will return a single string or comma separated strings.
    If the question is multiple choice, 单选, it will return a choice letter (e.g. A) and the corresponding text, separated by : .

    Args:
        item (dict): one problem element from the json file. 
    """
    
    #If fill-in-the-blank
    if item["type"] == "填空":
        #If there are multiple blanks
        if len(item["answer"]) > 1:
            return ", ".join(item["answer"])
        else:
            return item["answer"][0]
    #If multiple choice
    elif item["type"] == "单选":
        choice = item["answer"][0]
        return f"{choice}: {item['options'][choice]}"
    else:
        raise ValueError(f"Unknown question type: {item['type']}")
    
def create_full_user_prompt(item):
    """Function for creating the full user prompt for a given problem item.

    Args:
        item (dict): one problem element from the json file. 
    """
    answer_structured = structure_answer(item)
    return user_prompt_template.format(item['question'], answer_structured, item["analysis"])

def get_soluton_steps(item):
    # Get the full user prompt 
    full_user_prompt = create_full_user_prompt(item)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": full_user_prompt
            }
        ],
        temperature=0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content  # Assuming this returns the converted text

# Load JSON data
with open(original_question_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

counter = 0

# Iterate and convert each problem
for key, item in data.items():
    start_time = time.time()  # Capture the start time
    
    item['step_by_step_solution_text'] = get_soluton_steps(item)  # Add the converted question to the dictionary
    
    end_time = time.time()  # Capture the end time
    iter_time = end_time - start_time  # Calculate the time taken for this iteration
    
    print(f"The question {key} took {iter_time:.2f} seconds to convert")
    
    counter += 1  # Increment the counter

    # Save the progress at every 100 iterations
    if counter % 100 == 0:
        with open(annotated_question_file, 'w', encoding='utf-8') as temp_file:
            json.dump(data, temp_file, ensure_ascii=False, indent=2)
        print(f"Progress saved at iteration {counter}")

with open(annotated_question_file, 'w', encoding='utf-8') as temp_file:
    json.dump(data, temp_file, ensure_ascii=False, indent=2)
print(f"Progress saved at iteration {counter}")