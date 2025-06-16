import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",  # or use another variant
    google_api_key = GEMINI_API_KEY
)


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # or the appropriate model name
    google_api_key=GEMINI_API_KEY
)


st.header("Nihil's First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions",type="pdf")
    
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)
        
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size = 1000,
        chunk_overlap = 150, # bring me the last 150 chunks into this chunk also    
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)
    
    #Generating Embeddings 
    # embeddings = GoogleGenerativeAIEmbeddings(google_api_key = GEMINI_API_KEY)
        
    #Creating Vectore Store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    #Get User Question
    user_question = st.text_input("Type your question here.")

    #Do Similarity Search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)
        
        #Define the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=512
        )

        #chain -> take the question, get relevant document, pass it to the LLM, generate tne output
        #Output Results
        chain = load_qa_chain(llm,chain_type="stuff")
        with st.spinner("💭 Gemini is thinking..."):
            response = chain.run(input_documents=match, question=user_question)

        st.markdown(f"🧠 **Gemini says:**\n\n> {response}")
