#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


with open('.env', 'r') as file:
    content = file.read()

#print(content)


# In[2]:


#new_line = "\nOPENAI_API_KEY=......"

# Append to the .env file
#with open('.env', 'a') as file:
 #   file.write(new_line)


# In[3]:


#new_line = "\nGROQ_API_KEY=gsk_......"

# Append to the .env file
#with open('.env', 'a') as file:
 #  file.write(new_line)


# In[4]:


### Open AI API Key and Open Source models--Llama3,Gemma2,mistral--Groq

import os
from dotenv import load_dotenv
load_dotenv()

import openai
openai.api_key=os.getenv("OPENAI_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")
#groq_api_key


# In[6]:


pip install langchain-core==0.1.37 langchain==0.1.16 langchain-groq==0.0.2


# In[9]:


from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)
model


# In[ ]:


get_ipython().system('pip install --upgrade langchain langchain-core')


# In[10]:


pip install langchain_groq


# In[11]:


pip install langchain_core    


# #### when we provide a instruction to LLM model, use  system message.
# #### what human will ask to LLM will be given as human message.pass list of messages to LLM model and invoke it.

# In[13]:


from langchain_core.messages import HumanMessage,SystemMessage

messages=[
    SystemMessage(content="Translate the following from English to tamil"),
    HumanMessage(content="Hello How are you?")
]

result=model.invoke(messages)


# In[14]:


result


# #### to retrieve only response output, use string output parser

# In[15]:


from langchain_core.output_parsers import StrOutputParser
parser=StrOutputParser()
parser.invoke(result)


# ####  Using LCEL(lang chain expression language)- chain the components
# components - model, parser

# In[16]:


### Using LCEL- chain the components
chain=model|parser
chain.invoke(messages)


# #### using prompt template

# In[17]:


### Prompt Templates
from langchain_core.prompts import ChatPromptTemplate

generic_template="Trnaslate the following into {language}:"

prompt=ChatPromptTemplate.from_messages(
    [("system",generic_template),("user","{text}")]
)


# In[18]:


result=prompt.invoke({"language":"tamil","text":"Hello"})


# In[19]:


result.to_messages()


# In[20]:


##Chaining together components with LCEL
chain=prompt|model|parser
chain.invoke({"language":"tamil","text":"Hello"})

