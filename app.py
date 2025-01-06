# app.py
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class RAGChatbot:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Initialize language model (you can swap this with different models)
        model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.vectorstore = None
        self.chain = None

    def load_documents(self, pdf_files):
        """Load and process PDF documents"""
        documents = []
        for pdf in pdf_files:
            loader = PyPDFLoader(pdf)
            documents.extend(loader.load())

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )

        # Create conversation chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True,
        )

    def get_response(self, query: str) -> str:
        """Get response from the chatbot"""
        if not self.chain:
            return "Please load documents first."
        
        result = self.chain({"question": query})
        return result["answer"]

# Streamlit interface
def main():
    st.title("RAG Style Guide Assistant")
    
    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    
    # File uploader
    pdf_files = st.file_uploader(
        "Upload your style guide PDFs",
        type="pdf",
        accept_multiple_files=True
    )
    
    if pdf_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                st.session_state.chatbot.load_documents(pdf_files)
            st.success("Documents processed successfully!")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("What would you like to know about the style guide?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = st.session_state.chatbot.get_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

# To run:
# streamlit run app.py
