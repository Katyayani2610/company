import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

os.environ["OPENAI_API_TYPE"] = 'azure'
os.environ["AZURE_OPENAI_ENDPOINT"] = 'https://cxaicoe-openai.openai.azure.com/'
os.environ["OPENAI_API_KEY"] = 'f0dd5fb480d74684832ff376296b730b'

llm = AzureChatOpenAI(**{'deployment_name':'cx-aicoe-gpt4','openai_api_version':'2024-09-01-preview'},temperature=0.1)
import json 
import random
from langchain_core.prompts import PromptTemplate
import re

with open('sample_inputs_Final.json','r') as file:
    data = json.load(file)[:20]

prompt_inputs = [record['input'] for record in data] 
print("test")

def balance_brackets(json_string):
    # Stack to track opening brackets and their positions
    stack = []
    
    # Convert the string into a list of characters to modify it
    json_chars = list(json_string)
    
    # Iterate over the string to track open and close brackets
    for idx, char in enumerate(json_chars):
        if char == '{' or char == '[':
            stack.append((char, idx))  # Push opening brackets and their indices to the stack
        elif char == '}' or char == ']':
            if stack:
                last_open, last_idx = stack.pop()  # Pop the last opening bracket
                # If it's a mismatched pair, fix it
                if (char == '}' and last_open != '{') or (char == ']' and last_open != '['):
                    json_chars[last_idx] = ')'  # Replace last unmatched open bracket to correct type
                    stack.append((char, idx))   # Push this closing bracket back to stack
            else:
                # If it's an unmatched closing bracket, ignore it (but this could be handled as error)
                pass

    # Handle any unmatched opening brackets by adding the corresponding closing ones at the end
    while stack:
        last_open, last_idx = stack.pop()
        if last_open == '{':
            json_chars.append('}')  # Add closing brace at the end
        elif last_open == '[':
            json_chars.append(']')  # Add closing square bracket at the end
    
    # Return the balanced string
    return ''.join(json_chars)

output_format = '''{
        "user": "<Predicted user_input>",
        "Action": "Other",
        "attribute": "<Response as per user input in a string>"
        }'''
        
c=0
all_responses = []
for i in range(len(prompt_inputs)):
    chat_history_match = re.search(r'<chat_history>(.*?)</chat_history>', prompt_inputs[i], re.DOTALL)
    user_inputs_match = re.search(r'<user_input>(.*?)</user_input>', prompt_inputs[i], re.DOTALL)
    chat_history_raw = chat_history_match.group(1).strip() if chat_history_match else ""
    user_inputs_match = user_inputs_match.group(1).strip() if user_inputs_match else ""
    prompt_message = f''' 
    For the provided prompt, generate **15** alternative user inputs for each of the following scenarios:

    1. **Out-of-scope**: These are user inputs that are unrelated to the system's capabilities (e.g., if the system is designed for energy-related inquiries, user inputs like "Can you order me a pizza?" are out-of-scope).
    2. **Greet**: These are user inputs where the user greets the assistant (e.g., "Hi", "Hello", "Good morning").
    3. **Bye**: These are user inputs where the user says goodbye or ends the interaction (e.g., "Thanks", "I'm done", "Goodbye").
    4. **Say again**: These are user inputs where the user asks the assistant to repeat the last statement (e.g., "Come again", "What did you say?", "Can you repeat that?").
    5. **Hold on**: These are user inputs where the user asks the assistant to hold on for a moment (e.g., "Hold on a second", "Give me a moment", "Let me gather that information").

    For each user input, predict the corresponding **Action** and **Attribute** taking {chat_history_raw} into consideration.

    prompt: {prompt_inputs[i]}

    Output:
    [
        {output_format}
    ]

    Note: Please ensure that user inputs are diverse, realistic, and not repetitive. Return the output in the specified JSON format.     
    '''
    messages = [
                SystemMessage(content=prompt_message)
            ]
    try:
        response = llm.invoke(messages).content
        # print(response)
        response = balance_brackets(response)
        cleaned_response = re.sub(r'^```json\n|\n```$', '', response).strip()
        response_json = json.loads(cleaned_response)
        #print(response_json)
        user_inputs = [item['user'] for item in response_json]
        prompt_inputs_replaced = [prompt_inputs[i].replace(user_inputs_match, f"{user_input}") for user_input in user_inputs]
        print(prompt_inputs_replaced)
        datapoint_jsons = [{"input":prompt_input ,"output":[{"Action":"Other", "attribute":output_json['attribute']}]}
                           for (output_json, prompt_input) in zip(response_json, prompt_inputs_replaced)]
        all_responses += datapoint_jsons # Add the new responses to the overall list
    except json.JSONDecodeError:
        print(f"Error parsing response for prompt {i}.")
    c += 1
    print(c)

with open('responses_first_20.json', 'w') as outfile:
    json.dump(all_responses, outfile, indent=4)
