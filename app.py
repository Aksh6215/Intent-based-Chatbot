import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('wordnet')

# Load the final model
@st.cache_resource
def load_model():
    with open('final_model.pkl', 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict

model_dict = load_model()
tfidf = model_dict['tfidf']
model = model_dict['model']
df = model_dict['df']

# Preprocess text
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word.lower()) for word in tokens])

# Chatbot response function
def chatbot_response(user_input):
    processed_input = preprocess_text(user_input)
    input_tfidf = tfidf.transform([processed_input])
    intent = model.predict(input_tfidf)[0]
    response = df[df['tag'] == intent]['response'].sample().values[0]
    return response

# Streamlit UI
st.title("Cognitive Chat System")

st.write("Hi, Welcome to Intrix Chatbot! trained on specific intents to help you with your queries.")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Want to ask something:"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chatbot_response(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a brief explanation of the chatbot
st.sidebar.title("About")
st.sidebar.info(
    "This is an intent-based chatbot using NLP. "
    "It's trained on a custom dataset to understand user intents and provide appropriate responses. "
    "Feel free to start a conversation!"
)

# Optional: Add a button to clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []

# Add a "View History" button to display chat history
if st.sidebar.button("View History"):
    st.sidebar.write("### Chat History")
    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "Chatbot"
        st.sidebar.write(f"**{role}:** {message['content']}")



