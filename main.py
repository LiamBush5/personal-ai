# Import necessary libraries and modules
import os
import re
import time
import yaml
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chains import LLMChain, RetrievalQA, ConversationChain, ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_chat import message
import base64
import webbrowser

# Get the directory where the script is located
# Used to build the path to the configuration file
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to config.yaml
config_path = os.path.join(script_directory, 'config.yaml')

# Load configuration from a YAML file to setup the application
with open(config_path) as file:
    config = yaml.load(file, Loader=SafeLoader)

# Create an authenticator object to handle user login
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Initialize memory buffer for conversation if it doesn't exist in the session state
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(
        k=1, return_messages=True)

load_dotenv()
# PINECONE_API_ENV='asia-southeast1-gcp-free'
# PINECONE_API_KEY='96c1d3b3-440b-449d-a140-ffc0559c3197'

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
chat_history = []
# index_name = 'p'

# chat_history = []
# #pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
# index_name = 'rcl'
# # Load environment variables for Pinecone and OpenAI API keys
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

chat_history = []
index_name = 'policy-idx'

embedding = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY
)

# Initialize Pinecone
# Cache and initialize Pinecone to avoid reinitialization on every Streamlit run


@st.cache_resource
def start_pinecone():
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console)
    )

# Load Pinecone's existing index to perform similarity searches


@st.cache_resource
def load_pinecone_existing_index():
    docsearch = Pinecone.from_existing_index(
        index_name=index_name, embedding=embedding)
    return docsearch


# Functions to start Pinecone and load its existing index
start_pinecone()
vector_store = load_pinecone_existing_index()

# Function to remove HTML tags from a string using BeautifulSoup


def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text

# Function to further clean text, removing extra spaces, newlines, and brackets


def clean_text(text):
    cleaned_text = text.strip().strip("[]")
    cleaned_text = cleaned_text.replace("\n\n", "\n")
    cleaned_text = re.sub('<.*?>', '', cleaned_text)
    return cleaned_text

# Function to search for documents related to a given topic using Pinecone's similarity search


def get_docs(topic):
    docs = vector_store.similarity_search(
        topic,  # our search query
        k=8  # return 8 most relevant docs
    )
    return docs


# Function to generate a response based on the user's input topic
# It performs a similarity search, retrieves documents, and generates a response using a language model
def generate_response(topic):
    docs = vector_store.similarity_search(
        topic,  # our search query
        k=5 # return 8 most relevant docs
    )

    # Remove a doc if it has already been used
    used_sources = []
    sources = []
    for doc in docs:
        cleaned_doc_content = clean_text(doc.page_content)
        if cleaned_doc_content not in used_sources:
            used_sources.append(cleaned_doc_content)
            sources.append(doc)

    context = " ".join([doc.page_content for doc in sources])

    # Load previous chat history if available, else an empty list
    memory_variables = st.session_state.buffer_memory.load_memory_variables({
    })
    chat_history = memory_variables.get('history', [])
    # create a single input
    input = {'topic': topic, 'context': context,
             'chat_history': chat_history}
    output = chain_summarize.apply([input])

    # Extract the text from the dictionary
    text = output[0]['text']

    formatted_text = text

    # sources_list = []
    # sources_text = ""
    # source_count = 0

    # for i, source in enumerate(sources):
    #     cleaned_source = clean_text(source.page_content)
    #     source_count += 1
    #     sources_list.append(
    #         f"Source {source_count}:\n{cleaned_source}\n---------------------------------------------------------------------------")

    #     # Join all the cleaned sources into one string
    #     sources_text = "".join(sources_list)

    # a = sources_text
    # result = f"{formatted_text}\n\nSources:\n`{a}`"
    result = f"{formatted_text}"
    st.session_state.buffer_memory.save_context(
        {"input": str(topic)}, {"output": str(formatted_text)})

    # Also save the interaction in Streamlit's session state
    st.session_state['history'] = {"input": str(
        topic), "output": str(formatted_text)}
    return result
    
# script_template2 = PromptTemplate(
#     template="""
#     You are a chatbot trained on valiant solutions reusable content liabray, use use the chat histroy, souces, and topic to answer your questions, make sure your resompnses are accurate and no data is not made up. Disregard the History if its not relevant to the current topic below

#     History: {chat_history}
#     Sources: {context}
#     Topic: {topic}
#     """,
#     input_variables=["chat_history", "context", "topic"]
# )


# Create a prompt template to guide the AI in generating responses
# It includes an example and expects the AI to use a similar format
script_template2 = PromptTemplate(
    template="""
    You are a chatbot trained on Eric Chinn's Resume and work history also some personal info. Please respond like you are eric

    **Sources:** {context}  
    **Topic:** {topic}  
    """,
    input_variables=["chat_history", "context", "topic"]
)

# Create another template for cleaning up the AI's response, ensuring it's professional and in third person
clean_template = PromptTemplate(
    template=""" template="You are tasked with taking the response you just created and makeing sure that it is in the third person of valiant solutions and sounds profeshinal. DO NOT USE personal pronouns at all. Make it sound profeshinal: Context: {context} Blog post:""",
    input_variables=["context"]
)

# Initialize the OpenAI chat model with specific parameters, including API key and model name
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4',
    # model_name='gpt-3.5-turbo',
    temperature=0.25
)

# Create chains for summarizing and cleaning up the AI's responses
chain_summarize = LLMChain(
    llm=llm,
    prompt=script_template2,
    memory=st.session_state['buffer_memory'],
    verbose=True)

chain_clean = LLMChain(
    llm=llm, prompt=clean_template, verbose=True)

# Main function to run the Streamlit app


def runapp():
    # st.title("Valiant GPT Chatbot")
    st.title("Eric Chinn Career Persona")

    name, authentication_status, username = authenticator.login(
        'Login', 'main')

    if authentication_status:
        # Display logout button and welcome message
        authenticator.logout('Logout', 'main')
        st.write(f'Welcome *{name}*')

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # Placeholder for chat history
        chat_placeholder = st.empty()

        # user_input = st.text_input("Please type your question here")
        # user_input = st.text_input(
        #     "Please type your question here", key="user_query")
        # if st.button("Submit"):
        #     # st.session_state.user_query = user_input
        #     with st.spinner("Generating response..."):

        #         response = generate_response(st.session_state.user_query)
        #         st.session_state['chat_history'].append({'user': st.session_state.user_query, 'ai': response})
        #     # response = generate_response(user_input)
        #     # st.session_state['chat_history'].append(
        #     #     {'user': user_input, 'ai': response})
    
        #         # Display chat history in the placeholder
        #     st.markdown("### Chat History")
        #     for chat in reversed(st.session_state['chat_history']):
        #         message(f"**You:** {chat['user']}", is_user=True)
        #         message(f"**AI:** {chat['ai']}", is_user=False)

    
        user_input = st.text_input(
            "Please type your question here", key="user_query")
        if st.button("Submit"):
            # st.session_state.user_query = user_input
            with st.spinner("Generating response..."):

                # response = generate_response(st.session_state.user_query)
                # st.session_state['chat_history'].append({'user': st.session_state.user_query, 'ai': response})
                response = generate_response(user_input)
                st.session_state['chat_history'].append(
                     {'user': user_input, 'ai': response})

            # Display chat history in the placeholder
        st.markdown("### Chat History")
        for chat in reversed(st.session_state['chat_history']):
            message(f"**You:** {chat['user']}", is_user=True)
            message(f"**AI:** {chat['ai']}", is_user=False)

           # Check if user is authenticated
    elif authentication_status == False:
        # Display error message if login failed
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        # Ask for credentials if none provided
        st.warning('Please enter your username and password')


def main():
    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(
            k=1, return_messages=True)

    runapp()


# Entry point of the script
# Ensures that the main app is only run when the script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()
