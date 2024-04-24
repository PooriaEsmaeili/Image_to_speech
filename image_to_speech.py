#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dotenv import find_dotenv, load_dotenv
import os
from transformers import pipeline
import requests
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import PromptTemplate, LLMChain
from openai import OpenAI
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


# In[83]:


load_dotenv(find_dotenv())
HUGGINGACEHUB_API_TOKEN = os.getenv('HUGGINGACEHUB_API_TOKEN')


# In[3]:


#img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]['generated_text']

    return text


# In[38]:


#generate story
def generate_short_story(context):

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
   
    input_text = f"Generate a 20 words short story based on the following context: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(**input_ids, max_length=20, num_return_sequences=1, early_stopping=True)

    generated_story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_story


# In[81]:


#generate story
def generate_short_story2(context):

    pipe = pipeline("text2text-generation", model="google/flan-t5-base")
   
    input_text = f"Generate a short strory based on this context: {context}"

    outputs = pipe(input_text, max_length=20, num_return_sequences=1)

    generated_story = outputs[0]["generated_text"]
    return generated_story


# In[85]:


def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGACEHUB_API_TOKEN }"}
    payloads = {
        "inputs": message
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


# In[87]:


def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="*")

    st.header("Turn img into audio story")
    uploaded_file = st.file_uploader("choose an image", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_short_story2(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        
        st.audio("audio.flac")


# In[88]:


if __name__ == '__main__':
    main()
    

