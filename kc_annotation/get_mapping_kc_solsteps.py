import json
from openai import OpenAI
import re
import time
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--original_question_file', type=str, help='Path to the original question file')
parser.add_argument('--mapped_question_file', type=str, help='Path to the mapped question file')
args = parser.parse_args()

original_question_file = args.original_question_file
mapped_question_file = args.mapped_question_file

client = OpenAI()

system_prompt = """You are expert in Math education.  You are given a Math question, its solution steps, and its knowledge concept(s), which you have annotated earlier. Your task is to associate which solution steps require which knowledge concepts. Note that all solution steps and all knowledge concepts must be mapped, while many-to-many mapping is indeed possible. 

Each solution step and each knowledge concept is numbered. Your output should enumerate all solution step - knowledge concept pairs as numbers. 

Your output should meet all the below criteria: 
- Each solution step has to be paired. 
- Each knowledge concept has to be paired.
- Map a solution step with a knowledge concept only if they are relevant.
- Your pairs cannot contain artificial solution steps. For instance, If there are 4 solution steps, the pair "5-2" is indeed illegal.
- Your pairs cannot contain artificial knowledge concepts. For instance, If there are 3 knowledge concepts, the pair "3-5" is indeed illegal.

You will output solution step - knowledge concept pairs in a comma separated manner and in a single line. For example, if there are 4 solution steps and 5 knowledge concepts, one potential output could be the following: "1-1, 1-3, 1-5, 2-4, 3-2, 3-5, 4-2, 4-3, 4-5".

Observe that this output also meets all the criteria explained above. 

Now, for the given question, solution steps and knowledge concepts, please provide your mapping as the output."""

user_prompt_template="""Question: {}

Solution steps: {}
Knowledge concepts: {}"""

def get_structured_sol_steps(item):
    """Function for structuring solution steps for a given problem item.

    Args:
        item (dict): one problem element from the json file. 

    Returns:
        str: structured solution steps.
    """
    sol_steps = item["step_by_step_solution_list"]
    structured_sol_steps = ""
    for i, step in enumerate(sol_steps):
        structured_sol_steps += f"{i+1}) {step}\n"
    return structured_sol_steps

def get_structured_kcs(item):
    """Function for structuring knowledge concepts for a given problem item.

    Args:
        item (dict): one problem element from the json file. 

    Returns:
        str: structured knowledge concepts.
    """
    kcs = item['knowledge_concepts_list']
    structured_kcs = ""
    for i, kc in enumerate(kcs):
        structured_kcs += f"{i+1}) {kc}\n"
    return structured_kcs

def create_full_user_prompt(item):
    """Function for creating the full user prompt for a given problem item.

    Args:
        item (dict): one problem element from the json file. 
    """
    solution_steps = get_structured_sol_steps(item)
    knowledge_concepts = get_structured_kcs(item)
    return user_prompt_template.format(item['question'], solution_steps, knowledge_concepts)

def get_mapping(item):
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
        max_tokens=128,
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
    
    if not 'mapping_step_kc_gpt-4o' in item:
        item['mapping_step_kc_gpt-4o'] = get_mapping(item)  # Add the converted question to the dictionary
    
    end_time = time.time()  # Capture the end time
    iter_time = end_time - start_time  # Calculate the time taken for this iteration
    
    print(f"The question {key} took {iter_time:.2f} seconds to convert")
    #print("Generated answer:", item['mapping_step_kc_gpt-4-turbo'])
    
    counter += 1  # Increment the counter

    # Save the progress at every 100 iterations
    if counter % 20 == 0:
        with open(mapped_question_file, 'w', encoding='utf-8') as temp_file:
            json.dump(data, temp_file, ensure_ascii=False, indent=2)
        print(f"Progress saved at iteration {counter}")

with open(mapped_question_file, 'w', encoding='utf-8') as temp_file:
    json.dump(data, temp_file, ensure_ascii=False, indent=2)
print(f"Progress saved at iteration {counter}")   