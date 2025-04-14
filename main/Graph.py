from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from pydantic import BaseModel
from typing import Optional, Callable, Any

from langchain_openai import AzureChatOpenAI

import json
import re
from ast import literal_eval
import mysql.connector
from datetime import datetime

from utils import extract_json
from components.stack import prompt_stack,tool_stack
from components.llm import Llm

obj_llm=Llm()
model = obj_llm.model
model_mini =obj_llm.model_mini

class Graph(object):
    def __init__(self, state):
        self.state = state
        self.nodes = {
            "Process_identify": self.process_identifiaction_agent,
            "Verification": self.verification_agent,
            "OUTPUT_AGENT": self.output_agent,
            "prerequisites_check": self.prerequisites_check,
            "Recent_transactions": self.Recent_transactions,
            "Card_information": self.Card_initiate,
            "Tool call": self.tool_call,
            "Response Generator": self.Response_generator
        }

    def prerequisites_check(self):
        # print(self.state.verification,"-----------prerequisites_check")
        if self.state['verification']:
            return "Process_identify"
        else:
            return "Verification"

    def verification_agent(self):
        messages = [
            SystemMessage(content=prompt_stack['verification']['system']),
            HumanMessage(content=prompt_stack['verification']['input'].format(user_input=self.state['user_query'],
                                                                              chat_history="\n".join(
                                                                                  self.state['chat_history'][-5:]),
                                                                              extracted_entity=str(
                                                                                  self.state['node_data'])))
        ]
        response = model.invoke(messages)
        # print(response,"--------verification_agent")
        response = extract_json(response.content)
        print(response, "--------verification_agent", self.state['chat_history'])
        if response[0]['Action'] == 'Response':
            self.state['response'] = response[0]['attribute']
            if len(self.state['node_data']) == 0:
                self.state['node_data'] = response[0]['extracted_entity']
            else:
                for key in response[0]['extracted_entity']:
                    try:
                        if len(response[0]['extracted_entity'][key]) > 0:
                            self.state['node_data'][key] = response[0]['extracted_entity'][key]
                    except Exception as e:
                        print(e, "Exception in value assign verification agent")
            return "OUTPUT_AGENT"
        else:
            ## send a filller response
            self.state['tool'] = {"info": response[0]['attribute'], "agent": "verification"}
            return "Tool call"

    def tool_call(self):
        # print("tool call",self.state.tool['info']['name'])
        tool_response = tool_stack[self.state['tool']['info']['name']](**self.state['tool']['info']['parameters'])
        print(tool_response, "*****tool_response")
        if self.state['tool']['agent'] == "verification":
            if len(tool_response) == 1:
                self.state['tool_response'] = tool_response
                return "Response Generator"
            else:
                for key in tool_response:
                    try:
                        self.state[key] = tool_response[key]
                    except Exception as e:
                        print(e, "key to be inserted")
                        pass
                self.state['tool_response'] = {}
                self.state['tool_response']['comment'] = tool_response['comment']
                self.state['verification'] = True
                return "Response Generator"
        else:
            self.state['tool_response'] = tool_response
            return "Response Generator"

    def process_identifiaction_agent(self):
        messages = [
            SystemMessage(content=prompt_stack['process_identifiaction']['system']),
            HumanMessage(
                content=prompt_stack['process_identifiaction']['input'].format(user_input=self.state['user_query'],
                                                                               chat_history="\n".join(
                                                                                   self.state['chat_history'][-5:]),
                                                                               previous_process_state=self.state[
                                                                                   'previous_process_state']))
        ]
        response = model.invoke(messages)
        response = extract_json(response.content)
        print(response, "-------------process_identifiaction_agent")
        self.state['previous_process_state'] = response['Action']
        if response['Action'] == 'Other':
            self.state['response'] = response['attribute']
            return "OUTPUT_AGENT"
        else:
            return response['Action']

    def output_agent(self):
        # print("Marvin: " + self.state.response)
        self.state['chat_history'].append("User: " + self.state['user_query'])
        self.state['chat_history'].append("Marvin: " + self.state['response'])
        return 'END'

    def Recent_transactions(self):
        try:
            messages = [
                SystemMessage(content=prompt_stack['Recent_transactions']['system']),
                HumanMessage(
                    content=prompt_stack['Recent_transactions']['input'].format(user_input=self.state['user_query'],
                                                                                chat_history="\n".join(
                                                                                    self.state['chat_history'][-5:]),
                                                                                account_id=self.state['account_id']))
            ]
            # print(1111111111111)
            response = model.invoke(messages)
            response = extract_json(response.content)
        except Exception as e:
            print(e, "Exception in Recent_transactions")
        print(response, "------Recent_transactions")
        if response['Action'] == 'Response':
            self.state['response'] = response['attribute']
            return "OUTPUT_AGENT"
        else:
            # print(333333333333)
            self.state['tool'] = {"info": response['attribute'], "agent": "Recent_transactions"}
            return response['Action']

    def Card_initiate(self):
        self.state['response'] = "For all card related information, I have to transfer your request to our executives"
        return "OUTPUT_AGENT"

    def Response_generator(self):
        ## this will be LLM based
        # print(self.state.tool_response,"#########")
        try:
            messages = [
                SystemMessage(content=prompt_stack["Response Generator"][self.state['tool']['agent']]['system']),
                HumanMessage(content=prompt_stack["Response Generator"][self.state['tool']['agent']]['input'].format(
                    user_input=self.state['user_query'], chat_history="\n".join(
                        self.state['chat_history'][-5:]), result=str(self.state['tool_response'])))
            ]
            response = model_mini.invoke(messages)
            response = extract_json(response.content)
        except Exception as e:
            print(e, "Exception in Response_generator")
        self.state['response'] = response['attribute']
        return "OUTPUT_AGENT"
		
		