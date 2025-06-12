import os
import json
from glob import glob
import tiktoken
import difflib
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from os import listdir
from os.path import isfile, join
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
from config import configuration
import time
import numpy as np
from langchain.retrievers import BM25Retriever
from langchain_core.documents import Document
from chromadb import Documents, EmbeddingFunction, Embeddings

class isimilar_output(BaseModel):
    reshared_query: str = Field(description="return a standalone question from user follow up input")
    flag: str = Field(description="return string value True/False. if user input query was previously asked then True else False")
    response: str = Field(description="answer to the particular question present in conversation history if is_similar is True or return empty string,")

class AzureEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embeddingModel):
        self.embeddingModel = embeddingModel
    def __call__(self, input: Documents) -> Embeddings:
        embeddings_list = [self.embeddingModel.embed_query(text) for text in input]
        return embeddings_list
    def embed_query(self, text: str) -> List[float]:
        return self.embeddingModel.embed_query(text)  

    
class Retriever():
    def __init__(self, model_object, domain, demo_name, id=None):
        self.id = id
        self.fileTypes = [".txt", ".pdf", ".csv", ".docx"]
        self.embeddingModel = model_object.embeddingModel
        self.llm = model_object.llm
        self.domain = domain
        self.demo_name = demo_name
        

        prompt = '''
You are a AI-powered {domain} customer support assistant.You must answer only using the Retrieved documents for Customer question.Engage in an interactive Q&A format where the AI asks clarifying questions.you MUST NOT generate answer beyond {domain} context and You need to follow the following instructions:

**Important Instruction:**  
    1.If Customer asks question that is out of {domain} context then explicitly generate answer that the details are unavailable like "I don’t have much insight into this right now" and redirect the Customer to appropriate {domain} context for further assistance.
    2.If Customer asks question that is mix domain or topics then only consider {domain} context part of the question to generate the answer and indicating other context are unavailable like that "i don't have much informations explicitly about this at the moment".
    3.ONLY the answer to the Customer query in {domain} context or the greeting and closing messages and nothing else
    4.Avoids Hallucination.
    5.Do not add any new information. Strictly stick to the retrieved information.
    
    
**Instruction:** 
  **General Instruction:**
        1.Identify whether the Customer has a query in {domain} context or greeting message or closing message.
        2.Acknowledge the customer’s question with different **Customer_expression**.
        3.Ask follow-up questions if more details are needed to provide an accurate answer.
        4.Provide a clear, concise answer, ensuring it is easy to understand like a casual, conversational tone, like someone is explaining it naturally.
        5.Keep the conversation interactive
        4.If the user asks to summarize the conversation, take the **Conversation_history:** into account and give a concise summary of the previous conversations.
        5.Track the **Conversation_history:** to avoid repetition on same answer.
        6.You can not use words like retrieved documents or {domain} context while generating answer, else you would be PENALIZED.
        7.Examples can be used to construct the response but DO NOT respond with the exact messages from the Examples.
        8. If user has a greeting messages then generate a greetings messages
        9. If user has a closing messages then generate a closing messages

        
**Customer_expression**
    1. Use Casual expressions like 'hmm,' 'oh, I see,', and even humor or emotion where it fits.
    2. Use Pronunciation mistakes phonetically (like 'gonna' instead of 'going to,' 'lemme' instead of 'let me'), and include occasional word mix-ups or repeated phrases to make it sound authentic. 
    3. Use If something surprises or realization to you, react with curiosity like "Oh, wow!", "Really?", "Wait, what?","I didn’t know that!","No way!" "That’s crazy!". 
    4. Use If something understand by you, react like "Ah, okay","Got it.","I get what you mean.","Right, that makes sense","Oh, now I understand","I see what you're saying","That clears things up"          
    5. Use If something thoughtfulness or reflection by you, react like "Hmm, interesting","Let me think about that","That’s a good point","I never thought of it that way","You might be onto something","Hmm, I need to process that","I’m not sure about that"
    6. Use If something agreement or approval by you, react like "Sounds good to me","That works","I'm in","Sure, I’m down.","Absolutely","That’s perfect","Count me in","I’m all for it"
    7. Use If something is funny, laugh like "Hehe, that’s funny!","Haha!","Hehe!","Ahaha, that was a good one!","Hahaha, that’s hilarious!","Haha, wait, seriously?"
    8. Use If you're unsure, say "Hmm, I’m not so sure","I’m not convinced","I don’t think that’s right","Not quite","I don’t agree with that","I see your point, but…"
 
Examples for Greetings and closing message response:
 
**Greetings:**
    -Hey there! Looks like you’ve got a question—what’s on your mind?
    -What’s up, buddy? Got a question you want to go over?
    -Hey buddy! Anything I can do to help clear things up?
    -Hi buddy! What part of the lesson are you curious about?
    -Hey buddy! Got a question you need help with? I’m all ears.

**Closing Messages:**
    -Take care, buddy! Catch you later if you need help with anything else!
    -Alright, buddy, have a great day! Don’t hesitate to ask if you need anything!
    -See you later, buddy! I’m here if you need me again!
    -Alright, take it easy, buddy! I’ll be here if you have more questions later!
    -Catch you later, buddy! Don’t be shy if you need any help!    
        
     
**Question:**
{question}

**Retrieved documents**
{docs}

**Conversation_history:**
{chat_history}
'''
        prompt_template = PromptTemplate(template=prompt,input_variables=["question","docs","chat_history","domain"])
        self.chain = prompt_template | self.llm | StrOutputParser()

    def load_vectorstore(self):
        vector_store = Chroma(collection_name=self.domain,embedding_function=self.embeddingModel,persist_directory=f"./sampledb/{self.demo_name}")
        return vector_store

    def get_relevant_documents(self,query,k=10,rerank_top_k=3):
        chunks = []
        vectorstore = self.load_vectorstore()
        initial_docs=vectorstore.similarity_search_with_score(query)
        print("initial_docs",initial_docs)
        pairs = [Document(page_content=doc[0].page_content) for doc in initial_docs if doc[1]<=1]
        print("pairs",pairs)
        if len(pairs)>0:
            bm25_rank = BM25Retriever.from_documents(pairs,k=3)
            result = bm25_rank.invoke(query)
            return [result[i].page_content for i in range(len(result))]
        else:
            return []
        
    def rephrase_question(self, user_input, conversation_history):
        parser = JsonOutputParser(pydantic_object=isimilar_output)
        rephrase_question_prompt = PromptTemplate(
            input_variables=["user_input", "conversation_history"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template="""You are an AI assistant tasked with rephrasing user **Question** using the full **Conversation History** for context. Follow the rules below to ensure clarity and consistency.Do not expand any short abbreviations or shortened form to long form or full form (Example NLP to Natural language processing). 
 
**Important Instructions:**
        1.If the current question relies on prior context, rephrase it into a standalone question. Integrate relevant details from both earlier questions and answers to resolve ambiguous references (e.g., "it", "they") while preserving the original intent.
        2.If the user message is a casual acknowledgment (e.g., "okay", "thanks", "got it"), leave it unchanged. These are not rephrased.
        3.If the user gives a confirmation (e.g., "yes", "correct") in response to a prior question, rephrase it into a full statement that includes the relevant context.
        4.Do not rephrase greetings or closing messages (e.g., "Hi", "Hello", "Thank you").
        5. You must always return valid JSON fenced by a markdown code block. Do not return any additional text
 
**Question:**
{user_input}
 
**Conversation history:**
{conversation_history}

**format instruction:**
{format_instructions}
            """
        )
        self.rephrase_chain = rephrase_question_prompt|self.llm|parser
        rephrased_query = self.rephrase_chain.invoke({"conversation_history":conversation_history, "user_input":user_input})
        return rephrased_query

    def check_similarity(self, user_input, conversation_history):
        parser = JsonOutputParser(pydantic_object=isimilar_output)
        similarity_prompt_template = PromptTemplate(
         input_variables=["user_input", "conversation_history"],
         partial_variables={"format_instructions": parser.get_format_instructions()},
         template="""You are an AI assistant that checks if a question has already been asked in the **conversation history**.
         **If response to the current question is present in **conversation history** and **completely relevant** to the asked question, only then return "True", else return "False".**
         **If user asks a follow up question, if insufficient answer is present in the conversation history, instead of returning "True" and giving the same response from conversation history continuously, return "False".**
         If it returns "True", answer that particular question only using the **conversation history** and rephrase it according to the question asked and give the response using conversational fillers to make it sound more natural, else return empty as answer.
         **If a user asks to tell more about a topic or wants to know more about a topic even after giving all the details from conversation history, return "False"**
         **Always return "False" to greetings or if length of conversation history is empty or if its just an acknowledgement**.
         if a similar question is already present in conversation history add additional warm message like let's discuss it again and get things clarified, This time lets try to remove this prior query's doubts, I am happy to re-iterate this again to you, etc.
         Return output as a json having keys flag and response only.
        	
         output schema:
         {{flag: str,
         response: str}}
            
         question:
         {user_input}
         
         **conversation history**:
         {conversation_history}
         """
        )
        self.similarity_chain=similarity_prompt_template|self.llm|parser
        is_similar = self.similarity_chain.invoke({"conversation_history": conversation_history, "user_input": user_input})
        return is_similar

    def inference(self, query, chat_history):
        rephrased_query = self.rephrase_question(query,chat_history)
        print("rephrased_query",rephrased_query)
 
        if rephrased_query["response"]!="":
            return rephrased_query["response"]
        else:
            docs= self.get_relevant_documents(rephrased_query['reshared_query'])
            print("Documemt Retreived\n",docs)
            def format_docs(docs):
                return "\n\n".join(docs)
 
            result = self.chain.invoke({"question": rephrased_query['reshared_query'], "docs": format_docs(docs),"chat_history": chat_history, "domain": self.domain})
            return result
