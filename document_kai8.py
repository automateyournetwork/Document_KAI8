"""
Main application
"""
import os
import time
import logging
import pathlib
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
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

# Message classes
class Message:
    """
    Base message class
    """
    def __init__(self, content):
        self.content = content


class HumanMessage(Message):
    """
    Represents a message from the user.
    """


class AIMessage(Message):
    """
    Represents a message from the AI.
    """

class ChatWithFile:
    """
    Main class to handle the interface with the LLM
    """
    def __init__(self, file_path, file_type):
        """
        Perform initial parsing of the uploaded file and initialize the
        chat instance.

        :param file_path: Full path and name of uploaded file
        :param file_type: File extension determined after upload
        """
        self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:80/api/embeddings/generate")
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
        """
        Split the document pages into chunks based on similarity.
        """
        text_splitter = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type="percentile"
        )
        chunks = text_splitter.split_documents(pages)
        return chunks

    def simplify_metadata(self, doc):
        """
        If the provided doc contains a metadata dict, iterate over the
        metadata and ensure values are stored as strings.

        :param doc: Chunked document to process
        :return: Document with any metadata values cast to string
        """
        metadata = getattr(doc, "metadata", None)
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    metadata[key] = str(value)
        return doc

    def store_in_chroma(self, docs):
        """
        Store each document in Chroma.

        :param docs: Result of splitting pages into chunks
        :return: None
        """
        docs = [self.simplify_metadata(doc) for doc in docs]
        self.vectordb = Chroma.from_documents(docs, embedding=self.embedding_model)
        self.vectordb.persist()

    def initialize_llm_chains(self):
        """
        Initialize ConversationalRetrievalChain for each model.

        :return: Dictionary of ConversationalRetrievalChain instances
        """
        llm_chains = {}
        models = ["gemma", "aya", "llama3", "mistral", "mixtral", "qwen2", "phi3", "tinyllama"]

        def create_qa_chain(model):
            llm = Ollama(model=model, base_url=f"http://localhost:80/api/{model}/generate")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                self.vectordb.as_retriever(search_kwargs={"k": 10}),
                memory=self.memory
            )
            return qa_chain

        # Recreate the chains to maintain references
        for model in models:
            llm_chains[model] = create_qa_chain(model)
            time.sleep(5)
        return llm_chains

    def chat(self, question):
        """
        Main chat interface. Generate a list of queries to send to the LLM, then
        collect responses and append to the conversation_history instance
        attribute, for display after the chat completes.

        :param question: Initial question asked by the uploader
        :return: None
        """
        all_results = []
        
        for model, qa_chain in self.llm_chains.items():
            try:
                response = qa_chain.invoke(question)
                if response:
                    # Assuming the response is a dictionary, extract the 'answer' field
                    answer_text = response['answer'] if isinstance(response, dict) and 'answer' in response else str(response)
                    st.write(f"Query (Model: {model}): ", question)
                    st.write("Response: ", answer_text)
                    all_results.append(
                        {
                            "model": model,
                            "query": question,
                            "answer": answer_text
                        }
                    )
                else:
                    st.write(f"No response received for (Model: {model}): ", question)
            except Exception as e:
                st.write(f"Error with model {model}: {e}")

        if all_results:
            self.conversation_history.append(HumanMessage(content=question))
            for result in all_results:
                self.conversation_history.append(AIMessage(content=f"Model: {result['model']} - {result['answer']}"))

            return all_results

        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content="No answer available."))
        return {"answer": "No results were available to synthesize a response."}

def upload_and_handle_file():
    """
    Present the file upload context. After upload, determine the file extension
    and save the file. Set session state for the file path and type of file
    for use in the chat interface.

    :return: None
    """
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
    """
    Main chat interface - invoked after a file has been uploaded.

    :return: None
    """
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
