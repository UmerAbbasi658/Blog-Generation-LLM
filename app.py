# import streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain.llms import CTransformers 

# ## Function to get response
# def getLlamaresponse(input_text,no_words,blog_style):

#     ## Llama Model







# st.set_page_config(page_title='Generate Blogs',
#                    page_icon='Hello',
#                    layout='centered',
#                    initial_sidebar_state='collapsed')


# st.header("Generate my blogs")

# input_text=st.text_input("Enter Your Blog Topic")

# #Creating 2 more columns 2 additional fields
# col1,col2=st.columns([5,5])

# with col1:
#     no_words=st.text_input('No of words')
# with col2:
#     blog_style=st.selectbox('Writing the blog for',('Researchers','Data Scientist','Common Peoples'),index=0)
# submit=st.button('Generate The Blog')

# # Final Response
# if submit:
    # st.write(getLlamaresponse(input_text,no_words,blog_style))

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from constants import HUGGINGFACE_API_KEY

# Set up the Hugging Face API key
openai.api_key = HUGGINGFACE_API_KEY

# Load the tokenizer and model from Hugging Face
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def get_llama_response(input_text, no_words):
    # Encode the input text and generate a response
    inputs = tokenizer(input_text, return_tensors='pt')
    output = model.generate(inputs['input_ids'], max_length=int(no_words), num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

st.set_page_config(page_title='Generate Blogs',
                   page_icon='Hello',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate my blogs")

input_text = st.text_input("Enter Your Blog Topic")

# Creating 2 more columns with additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of words', '100')  # Default value set to '100'

with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common Peoples'), index=0)

submit = st.button('Generate The Blog')

# Final Response
if submit:
    if input_text and no_words:
        st.write(get_llama_response(input_text, no_words))
    else:
        st.warning("Please provide both the blog topic and the number of words.")
