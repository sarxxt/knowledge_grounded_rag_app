import re
import string
import logging
import traceback
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document

class DocumentGenerator:

    def __init__(self) -> None:
        pass

    def clean_data(self, data) -> str:
        """
        Cleans raw text data by removing unnecessary characters, extra spaces, and formatting.
    
        Parameters:
            - data (str): The raw text data to be cleaned.
    
        Returns:
            str: The cleaned text.
        """
        cleaned_data = data.replace("’", "").replace('\xa0', ' ').replace("'", "").replace("", "")
        cleaned_data = cleaned_data.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
        cleaned_data = re.sub(r"\B([A-Z][a-z])", r" \1", cleaned_data)
        cleaned_data = re.sub(r"[\([{})\]]", "", cleaned_data)
        cleaned_data = re.sub(' +', r' ', cleaned_data).lower()
        return cleaned_data

    def clean_merge_document(self, document_list, file_name) -> list:
        """
        Clean and merge similar pages from a list of documents and add metadata.

        Parameters:
        - document_list (List[Document]): List of documents containing page-wise content.
        - file_name (str): The name of the file from which the documents originated.

        Returns:
            List[Document]: List of page-wise cleaned documents with added metadata.
        """
        logging.info("Cleaning Documents")
        try:
            page_data = []
            page_documents = []

            for page_number, content in enumerate(document_list):
                cleaned_content = self.clean_data(content)
                page_documents.append(Document(page_content=cleaned_content, metadata={"page_number": page_number + 1, "file_name": file_name}))

            return page_documents

        except Exception as e:
            logging.error(f"Error while Cleaning Documents: {e} trace_back:{traceback.format_exc()}")
            raise Exception(f"Error: {e}")

    def generate_documents(self, file, file_name) -> list:
        """
        Generate and split documents from a PDF file using PyPDF2.

        Parameters:
        - file (str): The path of the PDF file to be processed.
        - file_name (str): The name of the file being processed.

        Returns:
        List[Document]: List of cleaned and split documents.
        """
        logging.info("Generating Documents")
        try:
            # Use PyPDF2 to read the PDF file
            reader = PdfReader(file)
            document_texts = [page.extract_text() for page in reader.pages if page.extract_text()]

            if not document_texts:
                raise Exception("No text found in the PDF file.")

            # Clean and merge document pages
            cleaned_pdf_documents = self.clean_merge_document(document_texts, file_name)

            # Split the cleaned documents into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=30)
            splitted_docs = text_splitter.split_documents(cleaned_pdf_documents)
            
            logging.info("Documents Generated Successfully")
        
            return splitted_docs

        except Exception as e:
            logging.error(f"Error while Generating Documents: {e} trace_back:{traceback.format_exc()}")
            raise Exception(f"Error: {e}")
