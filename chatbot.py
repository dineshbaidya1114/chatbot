import streamlit as st  #For userinterface(UI)
from dotenv import load_dotenv  #load.env into os.environ
import os   #interacts with environment vars
from langchain_groq import ChatGroq  #Groq LLM integration
from langchain.memory import ConversationBufferMemory ##Memory backend for chat
from langchain.chains import ConversationChain  #Chain that wires LLM+memory

#Load API Key
load_dotenv()  ## read .env file
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")  #setting up groq key

#Streamlit app setup
st.set_page_config(page_title="💬Conversational Q&A Chatbot")  #title in browser tab
st.title("💬Conversational Q&A Chatbot with message history")  #app header

## sidebar controls
model_name = st.sidebar.selectbox(
    "Select Groq model",
    ["gemma2-9b-it","deepseek-r1-distill-llama-70b","llama3-8b-8192"]

)
temperature = st.sidebar.slider( #fix the randomness of the response
    "Temperature",0.0,1.0,0.7
)

## what does it means setting temperature?
##When setting temperature in chatbot or language model development (like with GPT-based models), it refers to a parameter that controls the randomness or creativity of the model's responses. Here's a breakdown of what it means and how it works:

#What is Temperature? Temperature is a float value usually between 0.0 and 1.0, but some systems allow values above 1. It modulates the probability distribution of possible next tokens (words or characters) that the model might generate.

#Low Temperature (e.g., 0.0 – 0.3)--More deterministic and focused,Produces reliable, safe, and predictable responses. Ideal for factual, technical, or task-oriented bots (e.g., customer service, medical advice)
#Example: User: What is the capital of France?--Response (T=0.2): The capital of France is Paris.

#Medium Temperature (e.g., 0.5 – 0.7). Balances creativity and reliability. Generates slightly more varied and natural language. Suitable for conversational or general-purpose chatbots
#Example: User: Tell me something interesting about Paris--Response (T=0.7): Paris is known as the City of Light, not just for its nightlife, but also because it was one of the first cities to have street lighting.

#High Temperature (e.g., 0.8 – 1.2+). Highly creative and diverse, but less predictable. May generate unexpected or even incorrect responses. Useful for creative writing, brainstorming, or entertainment bots
#Example: User: Tell me a story set in Paris--Response (T=1.0): In the hidden catacombs beneath Paris, a jazz-playing ghost guards a time portal leading to the 1920s...

#Things to Keep in Mind
#Higher temperature = more variety, less coherence
#Lower temperature = more accuracy, less imagination
#Ideal temperature depends on use case and audience expectations

max_tokens = st.sidebar.slider(  #max response length
    "Max Tokens",50,300,150
)

#initialize memory and history

if "memory" not in st.session_state:  ##persists memory across reruns
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True  #return as list of messages, not one big string
    ) 

if "history" not in st.session_state: #stores role/content pairs display
    st.session_state.history=[]

# user input
user_input = st.chat_input("You: ") #clears itself on enter

if user_input: #append user turn to visible history
    st.session_state.history.append(("user",user_input))

    # instantiated a fresh llm for this turn
    llm=ChatGroq(
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_tokens
)

    #build conversationchain with our memory
    conv= ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose = True
)

    # get ai response(memory is updated internally)
    ai_response = conv.predict(input=user_input)

    #append assistent turn to visible history
    st.session_state.history.append({"assistant", ai_response})

#reder chat bubble
for role, text in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(text) #user style
    else:
        st.chat_message("assistant").write(text) #assistant style 