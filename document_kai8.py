import os
import time
import logging
import pathlib
import json
import requests
import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

logger = logging.getLogger(__name__)

FILE_LOADERS = {
    "csv": CSVLoader,
    "docx": Docx2txtLoader,
    "pdf": PyMuPDFLoader,
    "pptx": UnstructuredPowerPointLoader,
    "txt": TextLoader,
    "xlsx": UnstructuredExcelLoader,
}

ACCEPTED_FILE_TYPES = list(FILE_LOADERS)

class Message:
    def __init__(self, content):
        self.content = content

class HumanMessage(Message):
    pass

class AIMessage(Message):
    pass

@st.cache_resource
def load_model():
    try:
        with st.spinner("Downloading Instructor XL Embeddings Model locally....please be patient"):
            embedding_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"})
        return embedding_model
    except Exception as e:
        st.warning(f"CUDA not available or an error occurred: {e}. Falling back to CPU.")
        try:
            with st.spinner("Downloading Instructor XL Embeddings Model locally....please be patient (CPU fallback)"):
                embedding_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cpu"})
            return embedding_model
        except Exception as e:
            st.error(f"An error occurred while loading the model on CPU: {e}")
            return None

class ChatWithFile:
    def __init__(self, file_path, file_type):
        self.embedding_model = load_model()
        self.vectordb = None
        loader = FILE_LOADERS[file_type](file_path=file_path)
        pages = loader.load_and_split()
        docs = self.split_into_chunks(pages)
        self.store_in_chroma(docs)

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.llm_chains = self.initialize_llm_chains()

        self.conversation_history = []

    def split_into_chunks(self, pages):
        text_splitter = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type="percentile"
        )
        chunks = text_splitter.split_documents(pages)
        return chunks

    def simplify_metadata(self, doc):
        metadata = getattr(doc, "metadata", None)
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    metadata[key] = str(value)
        return doc

    def store_in_chroma(self, docs):
        docs = [self.simplify_metadata(doc) for doc in docs]
        self.vectordb = Chroma.from_documents(docs, embedding=self.embedding_model)

    def initialize_llm_chains(self):
        llm_chains = {}
        models = ["gemma", "aya", "llama3", "mistral", "wizardlm2", "qwen2", "phi3", "tinyllama", "openchat"]

        def create_qa_chain(model):
            llm = Ollama(model=model, base_url=f"http://localhost:80/api/{model}/generate")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                self.vectordb.as_retriever(search_kwargs={"k": 10}),
                memory=self.memory
            )
            return qa_chain

        for model in models:
            llm_chains[model] = create_qa_chain(model)
        return llm_chains

    def send_request(self, model, prompt):
        url = f"http://localhost:80/backend/{model}/generate"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": 0
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return response.json().get('response', '')
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"

    def chat(self, question):
        all_results = []
        response_placeholders = {}

        # Create a placeholder for each model's response
        for model in self.llm_chains.keys():
            response_placeholders[model] = st.empty()

        for model, qa_chain in self.llm_chains.items():
            try:
                response = qa_chain.invoke(question)
                if response:
                    answer_text = response['answer'] if isinstance(response, dict) and 'answer' in response else str(response)
                    response_placeholders[model].write(f"Model: {model}\nResponse: {answer_text}")
                    all_results.append(
                        {
                            "model": model,
                            "query": question,
                            "answer": answer_text
                        }
                    )
                else:
                    response_placeholders[model].write(f"Model: {model}\nNo response received.")
                    all_results.append(
                        {
                            "model": model,
                            "query": question,
                            "answer": "No response received."
                        }
                    )
            except Exception as e:
                response_placeholders[model].write(f"Model: {model}\nError: {e}")
                all_results.append(
                    {
                        "model": model,
                        "query": question,
                        "answer": f"Error: {e}"
                    }
                )

        consensus_prompt = (
            f"I am asking you to try and come to consensus with other LLMs on the answer to this question: "
            f"{question} Here are the answers from each LLM so far: {all_results}"
        )
        consensus_responses = []
        for model in self.llm_chains.keys():
            consensus_response = self.send_request(model, consensus_prompt)
            st.write(f"Consensus response from {model}: {consensus_response}")
            consensus_responses.append(consensus_response)

        final_consensus_prompt = (
            f"I am asking you to try and come to consensus with other LLMs on the answer to this question: "
            f"{question} Here are the consensus answers from each LLM so far: {consensus_responses}"
        )
        final_consensus_responses = []
        for model in self.llm_chains.keys():
            final_consensus_response = self.send_request(model, final_consensus_prompt)
            st.write(f"Final consensus response from {model}: {final_consensus_response}")
            final_consensus_responses.append(final_consensus_response)

        self.conversation_history.append(HumanMessage(content=question))
        for result in all_results:
            self.conversation_history.append(AIMessage(content=f"Model: {result['model']} - {result['answer']}"))

def upload_and_handle_file():
    st.title("Document KAI8 - Multi-AI Chat with Documents")
    uploaded_file = st.file_uploader(
        label=(
            f"Choose a {', '.join(ACCEPTED_FILE_TYPES[:-1]).upper()}, or "
            f"{ACCEPTED_FILE_TYPES[-1].upper()} file"
        ),
        type=ACCEPTED_FILE_TYPES
    )
    if uploaded_file:
        file_type = pathlib.Path(uploaded_file.name).suffix
        file_type = file_type.replace(".", "")

        if file_type:
            csv_pdf_txt_path = os.path.join("temp", uploaded_file.name)
            if not os.path.exists("temp"):
                os.makedirs("temp")
            with open(csv_pdf_txt_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state["file_path"] = csv_pdf_txt_path
            st.session_state["file_type"] = file_type
            st.success(f"{file_type.upper()} file uploaded successfully.")
            st.button(
                "Proceed to Chat",
                on_click=lambda: st.session_state.update({"page": 2})
            )
        else:
            st.error(
                f"Unsupported file type. Please upload a "
                f"{', '.join(ACCEPTED_FILE_TYPES[:-1]).upper()}, or "
                f"{ACCEPTED_FILE_TYPES[-1].upper()} file."
            )

def chat_interface():
    st.title("Document KAI8 - Multi-AI Chat with Documents")
    file_path = st.session_state.get("file_path")
    file_type = st.session_state.get("file_type")
    if not file_path or not os.path.exists(file_path):
        st.error("File missing. Please go back and upload a file.")
        return

    if "chat_instance" not in st.session_state:
        st.session_state["chat_instance"] = ChatWithFile(
            file_path=file_path,
            file_type=file_type
        )

    user_input = st.text_input("Ask a question about the document data:")
    if user_input and st.button("Send"):
        with st.spinner("Thinking..."):
            st.session_state["chat_instance"].chat(user_input)

if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state["page"] = 1

    if st.session_state["page"] == 1:
        upload_and_handle_file()
    elif st.session_state["page"] == 2:
        chat_interface()
