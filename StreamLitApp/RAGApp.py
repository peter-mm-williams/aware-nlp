import streamlit as st
import time
import rag_complete
import construct_thread

sub_reddit_list = ['BestBuyWorkers',
 'Bestbuy',
 'CVS',
 'Chase',
 'DisneyWorld',
 'Disneyland',
 'DollarTree',
 'FedEmployees',
 'Fedexers',
 'GameStop',
 'GeneralMotors',
 'KrakenSupport',
 'Lowes',
 'McDonaldsEmployees',
 'McLounge',
 'Panera',
 'PaneraEmployees',
 'RiteAid',
 'TalesFromYourBank',
 'Target',
 'TjMaxx',
 'UPSers',
 'WalmartEmployees',
 'WaltDisneyWorld',
 'cabincrewcareers',
 'cybersecurity',
 'disney',
 'fidelityinvestments',
 'nursing',
 'starbucks',
 'starbucksbaristas',
 'sysadmin',
 'walmart',
 'wholefoods']

llm_list = ['llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']


# Streamed response emulator
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# sidebar widgets

with st.sidebar:

    sub_reddit_select = st.selectbox("Select a sub-reddit:", options=sub_reddit_list)

    st.write("Retriever Configuration:")
    container1 = st.container(border=True)

    container1.write("Search parameters:")
    col1, col2, col3 = container1.columns(3)
    search_type = col1.selectbox("search_type", options=('similarity', 'mmr', 'similarity_score_threshold'))
    k_val = col2.number_input("k : ", value=3)
    score_threshold_val = col3.number_input('score_threshold : ', min_value=0.0, max_value=2.0)

    # st.write("Prompt Template for Generation:")
    llm_select  = st.selectbox("Select LLM for generation:", options=llm_list)
    groq_api_key = st.text_input("Groq API Key", type="password")

    # Check if API Key is provided
    if not groq_api_key:
        st.info("Please provide a Groq API Key to continue.")
        url = "https://console.groq.com/login" 
        st.write("[Get key here](%s)" % url)
        st.stop()
    container3 = st.container(border=True)
    generation_prompt_default = """Answer the following question based only on the provided context. If relevant context is not present reply with no relevant context present:

<context>
{context}
</context>

Question: {input}"""
    generation_prompt = container3.text_area("Prompt Template for Generation:", value=generation_prompt_default, height=300)

## Center elements

st.markdown("<h1 style='text-align: center;'> Retrival Agumented Generation </h1>", unsafe_allow_html=True)

# Set a LLM model
if "LLM_model" not in st.session_state:
    st.session_state["LLM_model"] = llm_select

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(message["user_prompt"])

    with st.container(border=True):
        st.write("Source Documents:")
        for docs in message["docs"]:
            st.write(docs["page_content"])
            st.write(docs["url"])

    with st.chat_message("assistant"):
        st.markdown(message["assistant_response"])
       

# Accept user input
doc_list = []
if prompt := st.chat_input(f"Input your query here (powered by {llm_select})"):
    # Add user message to chat history
    # st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

# Display retrieved documents

    with st.spinner('Retrieving documents...'):
        rep = rag_complete.BasicRAG(sub_reddit_select, 
                                prompt , 
                                llm_select, 
                                groq_api_key, 
                                'search_type', 
                                k_val, 
                                'score_threshold', 
                                generation_prompt_default)

        responses = rep.rag_response()

        with st.container(border=True):
            st.write("Source Documents:")
            for response in responses['context']:
                if response.metadata['aware_post_type']=='comment':
                    thread = construct_thread.ConsrtuctThread(rep.df_subreddit, response.metadata['reddit_name'])
                    st.write(response.page_content)
                    get_reddit_name = list(thread.get_thread().reddit_link_id)[0]
                    thread = construct_thread.ConsrtuctThread(rep.df_subreddit, get_reddit_name)
                    st.write(thread.get_url())
                    doc_list.append({"url" : thread.get_url(), "page_content": response.page_content})

                else:
                    thread = construct_thread.ConsrtuctThread(rep.df_subreddit, response.metadata['reddit_name'])
                    st.write(response.page_content)
                    st.write(thread.get_url())
                    doc_list.append({"url" : thread.get_url(), "page_content": response.page_content})
        

# Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(responses['answer']))
    st.session_state.messages.append({"user_prompt" : prompt, "assistant_response" : response, "docs": doc_list})