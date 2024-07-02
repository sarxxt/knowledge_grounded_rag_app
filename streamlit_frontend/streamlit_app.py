# Import necessary libraries
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import requests
import uuid
import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

BASE_URL = 'http://rag-backend-image:8000/'
# Response output
# Define a function for generating AI responses based on user input
def generate_response(user_input):
    url = f"{BASE_URL}/query/"
    headers = {
        'accept': 'application/json',
        'uuid': st.session_state['uuid'],
        'content-type': 'application/x-www-form-urlencoded',
    }

    params = {
        'query': user_input,
    }

    response = requests.post(url, params=params, headers=headers)
    return response.json()

# Define a function for taking user-provided prompts as input
def get_text():
    input_text = st.chat_input("Ask Anything.", key="input")
    return input_text

# Define a function to update the .env file with a new key-value pair
def update_env_file(key, value):
    with open('../.env', 'w') as f:
        f.write(f"\n{key}={value}")


# Set the page configuration
st.set_page_config(page_title="Chat Bot")

# Create the sidebar and set its title
st.sidebar.title("Insurance-BOT :speech_balloon:")

# Create an input box in the sidebar for the user to input a new environment variable
new_env_value = st.sidebar.text_input("Enter new environment variable value:")

if st.sidebar.button("Save to .env"):
    update_env_file('openai_api_key', new_env_value)
    st.sidebar.success("New environment variable saved!")

# Create an upload button in the sidebar for uploading a PDF file
uploaded_file = st.sidebar.file_uploader("Upload a file :inbox_tray:", type=["pdf"])
if uploaded_file is not None:
    if ('uploaded_file' not in st.session_state or uploaded_file.name != st.session_state['uploaded_file'].name):
        # Store the uploaded file in session state
        st.session_state['uploaded_file'] = uploaded_file
        # Define the URL for uploading PDF files
        url = f"{BASE_URL}/uploadpdf/"

        # Set request headers with a unique identifier (UUID)
        headers = {
            "uuid": st.session_state['uuid'],
        }

        # Prepare the file to be uploaded
        files = {"file": (uploaded_file.name, uploaded_file)}

        # Make a POST request to upload the file
        response = requests.post(url, headers=headers, files=files)
        print(response)

# Initialize empty lists for generated responses and past user questions
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# Generate a UUID if it doesn't exist in the session state
if 'uuid' not in st.session_state :
    st.session_state['uuid'] = str(uuid.uuid4())

# Create containers for input and response
response_container = st.container()
input_container = st.container()
styl = f"""
<style>
    .stTextInput {{
      position: fixed;
      bottom: 3rem;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

# Display the user input box
with input_container:
    user_input = get_text()

# Conditional display of AI generated responses based on user prompts
with response_container:
    if user_input:
        # Check if file is uploaded
        if 'uploaded_file' not in st.session_state:
            st.warning('Please upload a PDF first.')
        else:
            response = generate_response(user_input)

            # Store the user's input and AI-generated response
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            # Display the user's input as a message
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

            # Display the AI-generated response as a message
            message(st.session_state["generated"][i])

    if len(st.session_state['past']) == 0:
        st.title("Greetings!")
        st.markdown("Welcome to Insurance-Bot, ready to assist. Feel free to upload your PDF, and I'll do my best to provide helpful answers based on its content. Let's get started! :blush:")
