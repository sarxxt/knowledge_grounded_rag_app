import re, string
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.docstore.document import Document
import logging
import traceback
class DocumentGenerator:

    def __init__(self) -> None:
        pass

    def clean_data(self,data)-> str:
        """
        Cleans raw text data by removing unnecessary characters, extra spaces, and formatting.
    
        Parameters:
            - data (str): The raw text data to be cleaned.
    
        Returns:
            str: The cleaned text.
        """
        cleaned_data = data.replace("’ ","").replace('\xa0',' ').replace(" '","").replace("","")
        cleaned_data = cleaned_data.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
        cleaned_data = re.sub(r"\B([A-Z][a-z])", r" \1",cleaned_data)
        cleaned_data = re.sub(r"[\([{})\]]", "", cleaned_data)
        cleaned_data = re.sub(' +', r' ', cleaned_data).lower()
        return cleaned_data


    def clean_merge_document(self,document_list,file_name)-> list:
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
            # initializing empty lists
            page_data  = []
            page_documents = []
            previous_page = 0

            for document in document_list:
                
                page_number = document.metadata.get("page_number")
                
                if previous_page != page_number:
                    data = self.clean_data(' '.join(page_data))
                    page_documents.append(Document(page_content=data, metadata={"page_number":previous_page,"file_name": file_name}))
                    page_data.clear()
                    previous_page = page_number
                    
                else:
                    page_data.append(document.page_content)

            if page_data:
                data = self.clean_data(' '.join(page_data))
                page_documents.append(Document(page_content=data, metadata={"page_number":previous_page,"file_name": file_name}))
                page_data.clear()
                page_documents.pop(0)

            return page_documents
        
        except Exception as e:
            logging.error(f"Error while Cleaning Documents: {e} trace_back:{traceback.format_exc()}")
            raise Exception (f"Error: {e}")



    def generate_documents(self,file,file_name) -> list:
        """
        Generate and split documents from a PDF file.

        Parameters:
        - file (str): The path or content of the PDF file to be processed.
        - file_name (str): The name of the file being processed.

        Returns:
        List[Document]: List of cleaned and split documents.
        """
        logging.info("Generating Documents")
        try:
            pdf_loader = UnstructuredPDFLoader(file,mode="elements",strategy="ocr_only",infer_table_structure=True)
            pdf_text_documents = pdf_loader.load()
            cleaned_pdf_documents = self.clean_merge_document(pdf_text_documents,file_name)
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=30)
            Splitted_docs = text_splitter.split_documents(cleaned_pdf_documents)
            
            logging.info("Documents Generated Successfully")
        
            return Splitted_docs
        except Exception as e:
            logging.error(f"Error while Generating Documents: {e} trace_back:{traceback.format_exc()}")
            raise Exception (f"Error: {e}")

