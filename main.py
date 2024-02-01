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
    **Instructions:**
    - Begin with an cumulative summary addressing the query.
    - Respond with the file and section where the information is found.
    - Use direct quotes; do not summarize.
    - Use bullet points for listing information.
    - If the chat history or sources are not relevant, disregard them.
    - Insert a break between each cited file.
    - If another File is relevant respond with the file and section where the information is found, just like how you responded for thie first

    **Example Response:**

    **Question:** What is the GSA policy on Robotic Process Automation (RPA)?

    **Answer:**
    
    **Cumulative Summary:**
    The GSA’s policy on RPA is anchored in ensuring the robust security and integrity of communications and operations. It prescribes a comprehensive set of cybersecurity protocols and best practices to mitigate risks and enhance the security posture during UAS operations.

    **File:** Robotic-Process-Automation-(RPA)-Security-[CIO-IT-Security-19-97-Rev-3]-02-14-2023.pdf  
    **Section:** 3.5 Robotic Process Purpose Guidelines

    - "An important part of operating UAS is to ensure secure communications during all aspects of usage."
    - "UAS operators should evaluate the following cybersecurity best practices when conducting UAS operations."
    - "Ensure the data link supports an encryption algorithm for securing Wi-Fi communications."

    **File:** example_file.pdf  
    **Section:** 3.6 Example Section

    - "Important policy info directly quoted from the file."
    - "Another important quote that is relevant to the question."

    **History:** {chat_history}  
    **Sources:** {context}  
    **Topic:** {topic}  
    
    Possible Files:
    ['CIO IL-22-01 DevSecOps Model_Separation of Duties (03-10-2022) (1).pdf', 'IT_General_Rules_of_Behavior_CIO_21041B_CHGE_1_04-02-2019.pdf', 'CIO 21001N GSA Information Technology Security Policy 09-21-2022.pdf', 'Risk-Management-Strategy-(RMS)-[CIO-IT-Security-18-91-Rev-4]-06-28-2021docx.pdf', 'Conducting-Penetration-Test-Exercises-[CIO-IT-Security-11-51-Rev-6]-11-25-2022.pdf', 'Building Technologies Technical Reference Guide GSA v 20 (BTTRG) 06-11-2021.pdf', 'Information-Security-Continuous-Monitoring-Strategy-[CIO-IT-Security-12-66-Rev 4]-11-04-2022 (1).pdf', 'Robotic-Process-Automation-(RPA)-Security-[CIO-IT-Security-19-97-Rev-3]-02-14-2023.pdf', 'Identification_and_Authentication_(IA)_[CIO_IT_Security_01-01_Rev_6]_03-20-2019_Signed_BB.pdf', 'Managing-Enterprise-Cybersecurity-Risk-[CIO-IT-Security-06-30-Rev-24]-06-26-2023.pdf', 'Supply-Chain-Risk-Management-(SR)-Controls-[CIO-IT-Security-22-120]-04-15-2022docx.pdf', 'Lightweight-Security-Authorization-Process-(LATO) [CIO-IT-Security-14-68-Rev-7] 09-17-2021docx (1).pdf', 'Moderate-Impact-SaaS-Security-Authorization-Process-[CIO-IT-Security-18-88-Rev1]-03-31-2022docx.pdf', 'Low-Impact-SaaS-(LiSaaS)-Solutions-Authorization-Process-[16-75-Rev-5]-12-30-2022.pdf', 'FISMA-Implementation-Guide-[CIO-IT-Security-04-26-Rev3]-08-10-2022docx.pdf', 'Physical-and-Environmental-Protection-(PE)-[CIO-IT-Security-12-64-Rev-4]-07-08-2022docx (1).pdf', 'System-and-Information-Integrity-(SI)-[CIO-IT-Security-12-63-Rev-3]-09-30-2022.pdf', 'Security-and-Privacy-Awareness-and-Role-Based-Training-Program-[CIO-IT-Security-05-29-Rev-8]-05-17-2023 (1).pdf', 'Drones-Unmanned-Aircraft-Systems-(UAS)-Security-[CIO-IT-Security-20-104-Rev-1]-02-14-2023.pdf', 'Audit-and-Accountability-(AU)-[CIO-IT-Security-01-08-Rev-7]-02-21-2023.pdf', 'Configuration-Management-(CM)-[CIO-IT-Security-01-05-Rev-5]-03-01-2022docx.pdf', 'Securing-Mobile-Devices-and-Applications-[CIO-IT-Security-12-67-Rev-5]-06-16-2022docx.pdf', 'External-Information-System-Monitoring-[CIO-IT-Security-19-101-Rev-3]-03-31-2023.pdf', 'Contingency-Planning-(CP)-[CIO-IT-Security-06-29-Rev-6]-09-16-2022.pdf', 'Maintenance (MA) -[CIO-IT-Security-10-50-Rev-4]-11-15-2021docx (1).pdf', 'Plan-of-Action-and-Milestones-(POAM)-[CIO-IT-Security-09-44-Rev-8]-09-14-2022.pdf', 'Incident-Response-[CIO-IT-Security-01-02-Rev-19]-09-08-2022docx.pdf', 'IT-Security-Program-Management-Implementation (MIP) -Plan-(CIO-IT-Security-08-39-Rev-10]  01-30-2023.pdf', 'Access-Control-(AC)-[CIO-IT-Security-01-07-Rev-5]-08-18-2022.pdf', 'Web_Server_Log_Review_[CIO_IT_Security_08-41_Rev_4]_03_25_2020docx.pdf', 'Termination-and-Transfer-[CIO-IT-Security-03-23-Rev-6]-04-19-2022docx.pdf', 'OCISO-Cyber-Supply-Chain-Risk-Management-(C-SCRM)-Program-[CIO-IT-Security-21-117-Revsion-1]-03-07-2023.pdf', 'Protecting-CUI-Nonfederal-Systems-[CIO-IT-Security-21-112-Initial-Release]-05-27-2022.pdf', 'Firewall_and_Proxy_Change_Request_Process_[CIO_IT_Security_06-31_Rev_9]_12-22-2020 docx_.pdf', 'Key-Management-[CIO-IT Security-09-43-Revision 5]-04--6-2023.pdf', 'Annual-FISMA-and-Financial-Statements-Audit-Guide-[CIO-IT-Security-22-121, Revision 1]-05-15-2023.pdf', 'Security_and_Privacy_Requirements_for_IT_Acquisition_Efforts_[CIO_IT_Security_09-48_Rev_6]_04-15-2021 (1).pdf', 'Salesforce-Platform-Security-Implementation-[CIO-IT-Security-11-62-Rev 3]-03-01-2023.pdf', 'Vulnerability-Management-Process-[CIO-IT-Security-17-80-Rev-4]-03-13-2023.pdf', 'IT Security-and-Privacy-Awareness-and-Role-Based-Training-Program-[CIO-IT-Security-05-29-Rev-7]-09-29-2.pdf', 'Media-Protection-(MP)-[CIO-IT-Security-06-32-Rev-6]-11-18-2021docx.pdf', 'DevSecOps-Program-OCISO [CIO-IT-Security-19-102-Rev-2]-04-19-2023.pdf', 'Security-Engineering-Architectural-Reviews-[CIO-IT Security-19-95-Rev-1]-09-29-2022.pdf', 'Federalist-Security-Review-and-Approval-Process-[CIO-IT-Security-20-106-Revision-1]-03-27-2023.pdf', 'Physical-and-Environmental-Protection-(PE)-[CIO-IT-Security-12-64-Rev-4]-07-08-2022docx.pdf']
    ['CIO IL-22-01 DevSecOps Model_Separation of Duties (03-10-2022) (1).pdf', 'IT_General_Rules_of_Behavior_CIO_21041B_CHGE_1_04-02-2019.pdf', 'CIO 21001N GSA Information Technology Security Policy 09-21-2022.pdf', 'Risk-Management-Strategy-(RMS)-[CIO-IT-Security-18-91-Rev-4]-06-28-2021docx.pdf', 'Conducting-Penetration-Test-Exercises-[CIO-IT-Security-11-51-Rev-6]-11-25-2022.pdf', 'Building Technologies Technical Reference Guide GSA v 20 (BTTRG) 06-11-2021.pdf', 'Information-Security-Continuous-Monitoring-Strategy-[CIO-IT-Security-12-66-Rev 4]-11-04-2022 (1).pdf', 'Robotic-Process-Automation-(RPA)-Security-[CIO-IT-Security-19-97-Rev-3]-02-14-2023.pdf', 'Identification_and_Authentication_(IA)_[CIO_IT_Security_01-01_Rev_6]_03-20-2019_Signed_BB.pdf', 'Managing-Enterprise-Cybersecurity-Risk-[CIO-IT-Security-06-30-Rev-24]-06-26-2023.pdf', 'Supply-Chain-Risk-Management-(SR)-Controls-[CIO-IT-Security-22-120]-04-15-2022docx.pdf', 'Lightweight-Security-Authorization-Process-(LATO) [CIO-IT-Security-14-68-Rev-7] 09-17-2021docx (1).pdf', 'Moderate-Impact-SaaS-Security-Authorization-Process-[CIO-IT-Security-18-88-Rev1]-03-31-2022docx.pdf', 'Low-Impact-SaaS-(LiSaaS)-Solutions-Authorization-Process-[16-75-Rev-5]-12-30-2022.pdf', 'FISMA-Implementation-Guide-[CIO-IT-Security-04-26-Rev3]-08-10-2022docx.pdf', 'Physical-and-Environmental-Protection-(PE)-[CIO-IT-Security-12-64-Rev-4]-07-08-2022docx (1).pdf', 'System-and-Information-Integrity-(SI)-[CIO-IT-Security-12-63-Rev-3]-09-30-2022.pdf', 'Security-and-Privacy-Awareness-and-Role-Based-Training-Program-[CIO-IT-Security-05-29-Rev-8]-05-17-2023 (1).pdf', 'Drones-Unmanned-Aircraft-Systems-(UAS)-Security-[CIO-IT-Security-20-104-Rev-1]-02-14-2023.pdf', 'Audit-and-Accountability-(AU)-[CIO-IT-Security-01-08-Rev-7]-02-21-2023.pdf', 'Configuration-Management-(CM)-[CIO-IT-Security-01-05-Rev-5]-03-01-2022docx.pdf', 'Securing-Mobile-Devices-and-Applications-[CIO-IT-Security-12-67-Rev-5]-06-16-2022docx.pdf', 'External-Information-System-Monitoring-[CIO-IT-Security-19-101-Rev-3]-03-31-2023.pdf', 'Contingency-Planning-(CP)-[CIO-IT-Security-06-29-Rev-6]-09-16-2022.pdf', 'Maintenance (MA) -[CIO-IT-Security-10-50-Rev-4]-11-15-2021docx (1).pdf', 'Plan-of-Action-and-Milestones-(POAM)-[CIO-IT-Security-09-44-Rev-8]-09-14-2022.pdf', 'Incident-Response-[CIO-IT-Security-01-02-Rev-19]-09-08-2022docx.pdf', 'IT-Security-Program-Management-Implementation (MIP) -Plan-(CIO-IT-Security-08-39-Rev-10]  01-30-2023.pdf', 'Access-Control-(AC)-[CIO-IT-Security-01-07-Rev-5]-08-18-2022.pdf', 'Web_Server_Log_Review_[CIO_IT_Security_08-41_Rev_4]_03_25_2020docx.pdf', 'Termination-and-Transfer-[CIO-IT-Security-03-23-Rev-6]-04-19-2022docx.pdf', 'OCISO-Cyber-Supply-Chain-Risk-Management-(C-SCRM)-Program-[CIO-IT-Security-21-117-Revsion-1]-03-07-2023.pdf', 'Protecting-CUI-Nonfederal-Systems-[CIO-IT-Security-21-112-Initial-Release]-05-27-2022.pdf', 'Firewall_and_Proxy_Change_Request_Process_[CIO_IT_Security_06-31_Rev_9]_12-22-2020 docx_.pdf', 'Key-Management-[CIO-IT Security-09-43-Revision 5]-04--6-2023.pdf', 'Annual-FISMA-and-Financial-Statements-Audit-Guide-[CIO-IT-Security-22-121, Revision 1]-05-15-2023.pdf', 'Security_and_Privacy_Requirements_for_IT_Acquisition_Efforts_[CIO_IT_Security_09-48_Rev_6]_04-15-2021 (1).pdf', 'Salesforce-Platform-Security-Implementation-[CIO-IT-Security-11-62-Rev 3]-03-01-2023.pdf', 'Vulnerability-Management-Process-[CIO-IT-Security-17-80-Rev-4]-03-13-2023.pdf', 'IT Security-and-Privacy-Awareness-and-Role-Based-Training-Program-[CIO-IT-Security-05-29-Rev-7]-09-29-2.pdf', 'Media-Protection-(MP)-[CIO-IT-Security-06-32-Rev-6]-11-18-2021docx.pdf', 'DevSecOps-Program-OCISO [CIO-IT-Security-19-102-Rev-2]-04-19-2023.pdf', 'Security-Engineering-Architectural-Reviews-[CIO-IT Security-19-95-Rev-1]-09-29-2022.pdf', 'Federalist-Security-Review-and-Approval-Process-[CIO-IT-Security-20-106-Revision-1]-03-27-2023.pdf', 'Physical-and-Environmental-Protection-(PE)-[CIO-IT-Security-12-64-Rev-4]-07-08-2022docx.pdf']
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
    st.title("RVL2: GSA Polibot")

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
