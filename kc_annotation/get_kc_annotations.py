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

system_prompt = """You will be provided with a Math question, its final answer and its step by step solution. Your task is to provide the concise and comprehensive list of knowledge concepts in Math curriculum required to correctly answer the questions.

Please carefully follow the below instructions: 
- Provide multiple knowledge concepts only when it is actually needed. 
- Some questions require a figure, which you won't be provided. As the step-by-step solution is already provided, Use your judgement to infer which knowledge concept(s) might be needed.
- For a small set of solutions, their last step(s) might be missing due to limited token size. Use your judgement based on your input and your ability to infer how the solution would conclude. 
- Remember that knowledge concepts should be appropriate for Math curriculum. If annotated step-by-step solution involves more advanced techniques, use your judgment for more simplified alternatives.
- IMPORTANT: Provide only  the knowledge concepts, but nothing else. Separate them with a newline, and please don't use any enumeration or bullet points. In short, I want to be able to parse your listed knowledge concepts via splitting your output with \\n ."""

user_prompt_template="""Question: {}
Final Answer: {}
Step by Step Solution: {}"""

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
    return user_prompt_template.format(item['question'], answer_structured, item["step_by_step_solution_text"])

def get_kcs(item):
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
        max_tokens=256,
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
    
    item['knowledge_concepts_text'] = get_kcs(item)  # Add the converted question to the dictionary
    
    end_time = time.time()  # Capture the end time
    iter_time = end_time - start_time  # Calculate the time taken for this iteration
    
    print(f"The question {key} took {iter_time:.2f} seconds to convert")
    
    counter += 1  # Increment the counter

    # Save the progress at every 100 iterations
    if counter % 20 == 0:
        with open(annotated_question_file, 'w', encoding='utf-8') as temp_file:
            json.dump(data, temp_file, ensure_ascii=False, indent=2)
        print(f"Progress saved at iteration {counter}")

with open(annotated_question_file, 'w', encoding='utf-8') as temp_file:
    json.dump(data, temp_file, ensure_ascii=False, indent=2)
print(f"Progress saved at iteration {counter}")