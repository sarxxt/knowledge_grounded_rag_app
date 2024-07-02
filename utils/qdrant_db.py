# Import client library
from langchain.vectorstores import Qdrant
import logging 
import traceback

class Qdrant_DB():
    """
    Manages the storage and uploading of document vectors to a Qdrant database.

    This class is responsible for connecting to a Qdrant vector database, uploading document 
    vectors using a specified embedding model, and maintaining the collection within the Qdrant DB.
    """

    def __init__(self,embedding_model, collection_name):

        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.vector_store = None


    def upload_vectors(self,documents):
        """    
        Uploads document vectors to the in-memory Qdrant database collection.

        Processes a list of documents, generates embeddings using the assigned model, and 
        uploads these vectors to a specified Qdrant collection.
        """
        try:
            logging.info("Uploading Vectors in Qdrant DB")
            self.vector_store=Qdrant.from_documents(
                    documents=documents,
                    embedding=self.embedding_model, 
                    location=":memory:",
                    collection_name=self.collection_name,
                    m = 100,
                    ef_construction = 500
                )
            logging.info("Vectors Uploaded Successfully!")

        except Exception as e:
            error_msg = "Error While Uploading"
            logging.info(f" {error_msg} {e} trace_back:{traceback.format_exc()}")
            raise Exception (f"{error_msg}{e} ")



        
    def __del__(self):
        if self.vector_store:
            self.vector_store = None