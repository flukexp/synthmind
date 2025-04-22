import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

class DocumentIngestion:
    def __init__(self, data_dir="app/data/documents"):
        self.data_dir = data_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
    def load_documents(self):
        documents = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(self.data_dir, file))
                documents.extend(loader.load())
        return documents
    
    def process_documents(self):
        documents = self.load_documents()
        splits = self.text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        return vectorstore
    
    def get_or_create_vectorstore(self, force_reload=False):
        index_path = "app/data/faiss_index"
        if os.path.exists(index_path) and not force_reload:
            vectorstore = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            vectorstore = self.process_documents()
            os.makedirs(index_path, exist_ok=True)
            vectorstore.save_local(index_path)
        return vectorstore
