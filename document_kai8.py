import os
import pathlib
import logging
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
        selected_models = st.session_state.selected_models

        def create_qa_chain(model):
            llm = Ollama(model=model, base_url=f"http://localhost:80/api/{model}/generate")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                self.vectordb.as_retriever(search_kwargs={"k": 10}),
                memory=self.memory
            )
            return qa_chain

        for model in selected_models:
            if selected_models[model]:
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
                    response_placeholders[model].markdown(f"**Model: {model}**\n\nResponse: {answer_text}\n\n---")
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

        # Attempt to reach a consensus (optional)
        consensus_prompt = (
            f"Hello, esteemed models. I am seeking your collective expertise to reach a consensus on the following question: "
            f"{question}. Below are the individual responses from each model: {all_results}. "
            "Please review these responses carefully and provide a reasoned summary that attempts to align and synthesize the varied perspectives. "
            "Consider the strengths and weaknesses of each response, and aim to identify common themes or points of agreement."
            "Keep the original question in mind and do your best to come up with the best, most agreed upon, answer to that original question."
        )
        consensus_responses = []
        for model in self.llm_chains.keys():
            consensus_response = self.send_request(model, consensus_prompt)
            st.write(f"Consensus response from {model}: {consensus_response}")
            st.markdown("""---""")
            consensus_responses.append(consensus_response)

        # Final consensus prompt
        final_consensus_prompt = (
            f"Thank you for your thoughtful responses. Now, I ask you to further refine and come to a final consensus on the answer to this question: "
            f"{question}. Here are the preliminary consensus answers from each model so far: {consensus_responses}. "
            "Please critically evaluate these summaries, identify the most compelling arguments, and work towards a unified, well-supported final answer. "
            "Your final response should integrate the best elements of each perspective and resolve any remaining discrepancies."
            "Keep the original question in mind and do your best to come up with the best, most agreed upon, answer to that original question."
         )
        final_consensus_responses = []
        for model in self.llm_chains.keys():
            final_consensus_response = self.send_request(model, final_consensus_prompt)
            st.write(f"Final consensus response from {model}: {final_consensus_response}")
            st.markdown("""---""")
            final_consensus_responses.append(final_consensus_response)

        self.conversation_history.append(HumanMessage(content=question))
        for result in all_results:
            self.conversation_history.append(AIMessage(content=f"Model: {result['model']} - {result['answer']}"))

def model_selection():
    st.title("Select Models")
    all_models = ["gemma2", "aya", "llama3", "mistral", "wizardlm2", "qwen2", "phi3", "tinyllama", "openchat", "yi", "falcon2", "internlm2"]

    def select_all():
        for model in all_models:
            st.session_state.selected_models[model] = True

    def deselect_all():
        for model in all_models:
            st.session_state.selected_models[model] = False

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Select All Models'):
            select_all()
    with col2:
        if st.button('Deselect All Models'):
            deselect_all()

    col1, col2, col3 = st.columns(3)
    for idx, model in enumerate(all_models):
        col = [col1, col2, col3][idx % 3]
        with col:
            st.session_state.selected_models[model] = st.checkbox(model, value=st.session_state.selected_models[model], key=model)

    if st.button('Next'):
        st.session_state.page = 2
        st.rerun()

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
            if st.button("Proceed to Chat"):
                st.session_state.page = 3
                st.rerun()
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
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = {model: False for model in ["gemma2", "aya", "llama3", "mistral", "wizardlm2", "qwen2", "phi3", "tinyllama", "openchat", "yi", "falcon2", "internlm2"]}

    if st.session_state.page == 1:
        model_selection()
    elif st.session_state.page == 2:
        upload_and_handle_file()
    elif st.session_state.page == 3:
        chat_interface()
