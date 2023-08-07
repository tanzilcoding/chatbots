import os
import time
import openai
import pinecone
import streamlit as st
from streamlit_chat import message
import streamlit.components.v1 as components
from streamlit.components.v1 import html
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chains.question_answering import load_qa_chain

# cd /Users/macmini/Documents/Python/Streamlit/app_folder
# python3.11 -m venv venv
# source venv/bin/activate
# pip3.11 install --upgrade pip
# pip3.11 install openai
# pip3.11 install "pinecone-client"
# pip3.11 install langchain
# pip3.11 install streamlit
# pip3.11 install streamlit_chat
# pip3.11 install tiktoken
# pip3.10 install pipreqs
# pipreqs /Users/macmini/Documents/Python/Streamlit/app_folder
# deactivate
# =======================================================
# cd /Users/macmini/Documents/Python/Streamlit/app_folder
# source venv/bin/activate
# pip3.11 install --upgrade pip
# streamlit run streamlit_app.py
# =======================================================
# deactivate
# =======================================================

try:
    # Setting page title and header
    st.set_page_config(page_title="Trendlogic", page_icon=":robot_face:")
    st.markdown("<h1 style='text-align: center;'>Chatbot for Anders K. 😬</h1>", unsafe_allow_html=True)
    # system message to 'prime' the model
    primer = f"""You are Q&A bot. A highly intelligent system that answers
                user questions based on the information provided by the user above
                each question. If the information can not be found in the information
                provided by the user you truthfully say "I don't know".
                """

    # Set environment variables
    # Set org ID and API key
    # openai.organization = "<YOUR_OPENAI_ORG_ID>"
    # openai.organization = os.environ['openai_organization']
    # =======================================================
    pinecone_api_key = os.environ['pinecone_api_key']
    pinecone_environment = os.environ['pinecone_environment']
    OPENAI_API_KEY = os.environ['openai_api_key']
    openai.api_key = OPENAI_API_KEY
    index_name = os.environ['index_name']
    # ==================================================== #


    embed_model = "text-embedding-ada-002"

    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment  # find next to API key in console
    )

    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []
    if 'cost' not in st.session_state:
        st.session_state['cost'] = []
    if 'total_tokens' not in st.session_state:
        st.session_state['total_tokens'] = []
    if 'total_cost' not in st.session_state:
        st.session_state['total_cost'] = 0.0

    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    st.sidebar.title("Sidebar")
    model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    counter_placeholder = st.sidebar.empty()
    # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

    # Map model names to OpenAI model IDs
    if model_name == "GPT-3.5":
        model = "gpt-3.5-turbo"
    else:
        model = "gpt-4"

    # reset everything
    if clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state['number_tokens'] = []
        st.session_state['model_name'] = []
        st.session_state['cost'] = []
        st.session_state['total_cost'] = 0.0
        st.session_state['total_tokens'] = []
        counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


    # generate a response
    def generate_response(query):
        model_name = 'text-embedding-ada-002'

        embed = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=OPENAI_API_KEY
        )

        text_field = "text"
        # switch back to normal index for langchain
        index = pinecone.Index(index_name)

        vectorstore = Pinecone(
            index, embed.embed_query, text_field
        )

        # completion llm
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            # model_name='gpt-3.5-turbo',
            model_name=model,
            temperature=0.0,
            # verbose=True
        )

        messages = [
            SystemMessage(
                content=primer
            ),
            HumanMessage(
                content=query
                # content="Please give me a detailed answer to my question."
            ),
        ]

        llm(messages)

        docs = vectorstore.similarity_search(
            query,  # our search query
            k=5  # return 5 most relevant docs
        )

        # st.sidebar.text(docs)

        chain = load_qa_chain(llm, chain_type="stuff",
                                # verbose=True
                                )

        question = query
        raw_answer = chain.run(input_documents=docs, question=question)

        # st.sidebar.text(raw_answer)

        response = raw_answer.strip()

        st.session_state['messages'].append({"role": "user", "content": query})
        st.session_state['messages'].append({"role": "assistant", "content": response})

        # print(st.session_state['messages'])
        # total_tokens = completion.usage.total_tokens
        # prompt_tokens = completion.usage.prompt_tokens
        # completion_tokens = completion.usage.completion_tokens
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        return response, total_tokens, prompt_tokens, completion_tokens


    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['model_name'].append(model_name)
            st.session_state['total_tokens'].append(total_tokens)

            # from https://openai.com/pricing#language-models
            if model_name == "GPT-3.5":
                cost = total_tokens * 0.002 / 1000
            else:
                cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

            st.session_state['cost'].append(cost)
            st.session_state['total_cost'] += cost

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                # st.write(f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
                # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
except Exception as e:
    error_message = ''
    # st.text('Hello World')
    st.error('An error has occurred. Please try again.', icon="🚨")
    # Just print(e) is cleaner and more likely what you want,
    # but if you insist on printing message specifically whenever possible...
    if hasattr(e, 'message'):
        error_message = e.message
    else:
        error_message = e
    st.error('ERROR MESSAGE: {}'.format(error_message))
