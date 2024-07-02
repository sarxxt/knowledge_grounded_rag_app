from fastapi import FastAPI, File, UploadFile, Header,HTTPException,Response,status
from fastapi.middleware.cors import CORSMiddleware
import tempfile,shutil
from utils import (Qdrant_DB,EmbeddingModel,DocumentGenerator,gpt_chain)
import logging
import traceback

app = FastAPI()


logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',force=True)
# Setup logging
logging.basicConfig(level=logging.INFO)

# CORS settings
origins = ["*"]  # Update based on your specific needs

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],  # Limited to POST requests
    allow_headers=["Authorization", "Content-Type"],
)

embedding_model = EmbeddingModel()
document_generator = DocumentGenerator()
VECTOR_STORES = {}


@app.post("/uploadpdf/")
async def upload_pdf_endpiont(file: UploadFile = File(...),uuid: str = Header(...)):
    """
    Handles the upload and vectorization of a PDF file for a given UUID.

    Args:
    - file: UploadFile to be processed.
    - uuid: Header identifier for creating/updating vector store.

    Returns:
    - Success message upon successful upload and processing.

    """

    if file and file.filename.endswith('.pdf'):

        try:
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:

                shutil.copyfileobj(file.file, temp_file)
                file_name = file.filename.split('.')[0]
                docs = document_generator.generate_documents(file=str(temp_file.name),file_name=file_name)

                temp_file.flush()


            if uuid in VECTOR_STORES.keys():
                vector_store = VECTOR_STORES.get(uuid)
                del vector_store

            collection_name = f"temp_{uuid}"
            qdrant = Qdrant_DB(embedding_model,collection_name)
            vector_store = qdrant.upload_vectors(docs)
            VECTOR_STORES.update({uuid:qdrant})

            logging.info("File Uploaded Successfully")
            return Response(content="File Uploaded Successfully",status_code=status.HTTP_200_OK)

        except Exception as e:
            logging.error(f"Error occur: {e} trace_back:{traceback.format_exc()}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Something Went Wrong! Please Try Again.")

    else:
        error_msg = "Invalid File Type Only PDF"
        logging.error(error_msg)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,detail= error_msg)


@app.post("/query/")
async def query_endpoint(query: str,uuid: str = Header(...)):
    """
    Executes a query against a vector store identified by UUID.

    Args:
    - query: A string containing the query.
    - uuid: Header identifier for the associated vector store.

    Returns:
    - Response from LLM.
    """
    vector_store = VECTOR_STORES.get(uuid)
    if vector_store is not None:
        try:

            result = gpt_chain(vector_store.vector_store,query)
            return result

        except Exception as e:
            logging.error( f"Error occured while Quering{e} trace_back:{traceback.format_exc()}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Something Went Wrong! Please Try Again.")

    else:
        error_msg = "User Not Found"
        logging.error(error_msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail=error_msg)


@app.post("/clean_db/")
async def clean_db_endpoint(uuid :str = Header(...)):
    """
    Removes a vector store associated with the provided UUID.

    Args:
    - uuid: Header identifier for the vector store to be deleted.

    Returns:
    - A confirmation message that the user has been deleted.
    """
    if uuid in VECTOR_STORES.keys():
        try:

                del VECTOR_STORES[uuid]

                logging.info(f"User Deleted with uuid {uuid}")
                return Response(content="User Deleted",status_code=status.HTTP_200_OK)


        except Exception as e:
            logging.error(f"Error During Cleaning DB {e} trace_back:{traceback.format_exc()}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Something Went Wrong! Please Try Again.")
    else:
        logging.error(f"User with uuid {uuid} not found")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User Not Found.")



@app.get("/")
async def health_check_enpoint():
    """
    Provides a health check message for the service.

    Returns:
    - Response: A response object with the content "Hi, I am Healthy RAG".
    """
    return Response(content="Hi, I am Healthy RAG",status_code=status.HTTP_200_OK)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app", host="0.0.0.0", port=8000, reload=True)
