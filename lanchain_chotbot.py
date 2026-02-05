
import os
import shutil
from typing import Optional, Dict, Any, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAGServiceModern:
    
    def __init__(self, vector_db_path: str = "vector_db", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):

        self.vector_db_path = vector_db_path
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.llm = OllamaLLM(model="llama2")
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.chat_history = ChatMessageHistory()
        self.rag_chain = None
        
        if not os.path.exists(self.vector_db_path):
            os.makedirs(self.vector_db_path)
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:

        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            self.vectorstore = FAISS.from_documents(documents, self.embedding_model)
            self.vectorstore.save_local(self.vector_db_path)
            
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            self._create_rag_chain()
            
            return {
                "status": "success",
                "message": f"PDF processed successfully. Loaded {len(documents)} pages.",
                "num_documents": len(documents)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing PDF: {str(e)}"
            }
    
    def _create_rag_chain(self):
        if not self.retriever:
            return
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context. 
Use the following pieces of context to answer the question. 
If you don't know the answer based on the context, say so - don't make up information.

Context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self.chat_history.messages
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def load_existing_vectorstore(self) -> Dict[str, Any]:

        try:
            if not os.path.exists(self.vector_db_path):
                return {
                    "status": "error",
                    "message": "No existing vector store found"
                }
            
            self.vectorstore = FAISS.load_local(
                self.vector_db_path, 
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            self._create_rag_chain()
            
            return {
                "status": "success",
                "message": "Vector store loaded successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error loading vector store: {str(e)}"
            }
    
    def ask_question(self, question: str) -> Dict[str, Any]:

        if self.rag_chain is None:
            return {
                "status": "error",
                "message": "Please upload and process a PDF first"
            }
        
        try:
            answer = self.rag_chain.invoke(question)
            
            self.chat_history.add_user_message(question)
            self.chat_history.add_ai_message(answer)
            
            return {
                "status": "success",
                "question": question,
                "answer": answer
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing question: {str(e)}"
            }
    
    def get_chat_history(self) -> List[Dict[str, str]]:

        chat_history = []
        messages = self.chat_history.messages
        
        for msg in messages:
            chat_history.append({
                "type": msg.type,
                "content": msg.content
            })
        return chat_history
    
    def clear_session(self) -> Dict[str, Any]:

        try:
            self.chat_history.clear()
            
            self.retriever = None
            self.vectorstore = None
            self.rag_chain = None
            
            if os.path.exists(self.vector_db_path):
                shutil.rmtree(self.vector_db_path)
                os.makedirs(self.vector_db_path)
            
            return {
                "status": "success",
                "message": "Session cleared successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error clearing session: {str(e)}"
            }
    
    def delete_pdf(self, pdf_path: str) -> Dict[str, Any]:

        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                return {
                    "status": "success",
                    "message": "PDF deleted successfully"
                }
            return {
                "status": "warning",
                "message": "PDF file not found"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error deleting PDF: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:

        return {
            "vectorstore_loaded": self.vectorstore is not None,
            "retriever_ready": self.retriever is not None,
            "qa_chain_ready": self.rag_chain is not None,
            "chat_history_length": len(self.chat_history.messages)
        }
        
        
        
        
        
        