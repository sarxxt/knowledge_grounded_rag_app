from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import logging
import traceback
from typing import Dict
from utils import MilvusManager, DocumentGenerator  # Assuming this is in your utils.py file
from pymilvus import FieldSchema, DataType
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
import uuid
# from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from langchain.prompts import PromptTemplate

# Initialize OpenAI LLM via LangChain

# FastAPI app
app = FastAPI()

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

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

# Initialize required classes
# embedding_model = EmbeddingModel()
document_generator = DocumentGenerator()
        # Initialize OpenAI embeddings via LangChain
OPENAI_API_KEY = "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"
llm = ChatOpenAI(model_name='gpt-4o-mini',temperature=0.4, openai_api_key=OPENAI_API_KEY)
openai_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# embeddings.embed_documents(texts)
# Instantiate the MilvusManager class globally
milvus_manager = MilvusManager(host="127.0.0.1", port="19530")

@app.get("/create-user-token")
async def create_token():
    """
    Creates a user token (UUID) and sets up a Milvus collection schema for the user.
    
    Returns:
    - token: The unique token (UUID) assigned to the user.
    """
    try:
        # Define Milvus collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),  # OpenAI embeddings
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),  # Filename
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000)  # Adding text content field
        ]

        # Generate a unique UUID
        tenant_id = str(uuid.uuid4())

        # Create a collection for the UUID in Milvus
        token = milvus_manager.create_tenant_collection(tenant_id=tenant_id, fields=fields)

        return {"token": token}

    except Exception as e:
        return {"error": f"Failed to create user token. Error: {str(e)}"}

@app.post("/upload-pdf")
async def upload_pdf_endpoint(file: UploadFile = File(...), uuid: str = Header(...)):
    """
    Handles the upload and vectorization of a PDF file for a given UUID.

    Args:
    - file: UploadFile to be processed.
    - uuid: Header identifier for creating/updating vector store.

    Returns:
    - Success message upon successful upload and processing.
    """
    if not file.filename.endswith('.pdf'):
        error_msg = "Invalid file type. Only PDF is accepted."
        logging.error(error_msg)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=error_msg)

    try:
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            # Save the uploaded file to a temporary location
            shutil.copyfileobj(file.file, temp_file)
            file_name = file.filename.split('.')[0]

            # Generate documents (using your custom document generator logic)
            docs = document_generator.generate_documents(file=str(temp_file.name), file_name=file_name)

            # Ensure that data is written before proceeding
            temp_file.flush()



        logging.info(f"file name : {file_name}")

        # Extract text and generate embeddings
        texts = [doc.page_content for doc in docs]  # Extracting the text from the Document instances
        embedded_docs = openai_embeddings.embed_documents(texts)

        if len(embedded_docs) != len(texts):
            raise ValueError(f"Mismatch between number of embeddings and texts. {len(embedded_docs)}, {len(texts)}")
        # Check if the filename already exists in the tenant's collection
        if milvus_manager.filename_exists(tenant_id=uuid, filename=file_name):
            error_msg = f"Filename '{file_name}' already exists. Upload aborted."
            return Response(content=error_msg, status_code=status.HTTP_200_OK)

            # logging.error(error_msg)
            # raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error_msg)

        # Prepare data for insertion
        data_to_insert = [
            # Embeddings as vectors
            embedded_docs,
            # Repeating filename for each embedding
            [file_name] * len(embedded_docs),
            # The actual text content to be inserted
            texts
        ]

        # Ensure all data lists are the same length before inserting
        if any(len(lst) != len(embedded_docs) for lst in data_to_insert):
            raise ValueError("Data fields are not of the same length.")

        milvus_manager.insert_data(tenant_id=uuid, data=data_to_insert)

  
        logging.info(f"File {file_name} uploaded and processed successfully for UUID {uuid}.")
        return Response(content=f"File {file_name} uploaded and processed successfully.", status_code=status.HTTP_200_OK)

    except Exception as e:
        logging.error(f"Error occurred during processing: {e}, traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Something went wrong! Please try again.")

async def call_openai_llm_via_langchain(query: str, context: str) -> str:
    """
    Calls the OpenAI LLM via LangChain to generate a response based on the query and search context.

    Args:
    - query: The original query.
    - context: The search result context to pass to the LLM.

    Returns:
    - The generated response from the LLM.
    """
    try:
        # Create a combined prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="Context:\n{context}\n\nQuestion:\n{query}"
        )

        # Format the prompt using the context and query
        # prompt_text = prompt_template.format(context=context, query=query)
        prompt_text = f"Context:\n{context}\n\nQuestion:\n{query}"

        # # Generate a response using the LLM
        # response = llm(prompt_text)

        # Wrap the input in a HumanMessage object (required by ChatOpenAI)
        messages = [HumanMessage(content=prompt_text)]

        # Generate a response using the ChatOpenAI model
        response = llm(messages)  # llm should be a ChatOpenAI instance

        # Return the response as plain text
        return response.content.strip()

    except Exception as e:
        logging.error(f"Error occurred while calling OpenAI LLM: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LLM call failed.")


@app.post("/query")
async def query_documents(query: str, uuid: str = Header(...), top_k: int = 5):
    """
    Queries the document embeddings in the Milvus vector store using OpenAI embeddings and returns the closest matches.

    Args:
    - query: The query text.
    - uuid: The tenant UUID for identifying the collection.
    - top_k: Number of top results to return.

    Returns:
    - A list of documents most similar to the query, including text and filename.
    """
    try:
        # Embed the query using OpenAI
        query_embedding = openai_embeddings.embed_query(query)

        # Define search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        # Perform the search in Milvus
        results = milvus_manager.search(tenant_id=uuid, query_vectors=[query_embedding], top_k=top_k, search_params=search_params)
        # Extract the text from the search results to build the context
        context = "\n".join([result['text'] for result in results])

        # Call the LLM using the query and context via LangChain
        llm_response = await call_openai_llm_via_langchain(query, context)

        # Return the formatted results including text, filename, and LLM response
        return {
            "llm_response": llm_response,
            "relevant_context_from_vector_db": results
        }

        # Return the formatted results including text and filename
        # return {"results": results}

    except Exception as e:
        logging.error(f"Error occurred during query: {e}, traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error occurred during query.")


from typing import List, Optional

@app.post("/query-with-selected-files")
async def query_documents_with_file(
    query: str, 
    uuid: str = Header(...), 
    top_k: int = 5, 
    file_names: Optional[List[str]] = None  # New optional parameter for filenames
):
    """
    Queries the document embeddings in the Milvus vector store using OpenAI embeddings and returns the closest matches.

    Args:
    - query: The query text.
    - uuid: The tenant UUID for identifying the collection.
    - top_k: Number of top results to return.
    - file_names: Optional list of filenames to filter the query.

    Returns:
    - A list of documents most similar to the query, including text and filename.
    """
    try:
        # Embed the query using OpenAI
        query_embedding = openai_embeddings.embed_query(query)

        # Define search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        # If file_names is provided, filter the documents by filename
        if file_names:
            # Generate an expression to filter by the provided filenames
            file_filter_expr = f"filename in {file_names}"
            results = milvus_manager.search_with_filter(
                tenant_id=uuid, query_vectors=[query_embedding], top_k=top_k, 
                search_params=search_params, filter_expr=file_filter_expr
            )
        else:
            # No filenames provided, query all documents
            results = milvus_manager.search(
                tenant_id=uuid, query_vectors=[query_embedding], top_k=top_k, search_params=search_params
            )

        # Extract the text from the search results to build the context
        context = "\n".join([result['text'] for result in results])

        # Call the LLM using the query and context via LangChain
        llm_response = await call_openai_llm_via_langchain(query, context)

        # Return the formatted results including text, filename, and LLM response
        return {
            "llm_response": llm_response,
            "relevant_context_from_vector_db": results
        }

    except Exception as e:
        logging.error(f"Error occurred during query: {e}, traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error occurred during query.")

@app.get("/list-files/")
async def list_files(uuid: str = Header(...)):
    """
    Lists all files in the Milvus collection based on the tenant UUID.

    Args:
    - uuid: The tenant UUID for identifying the collection.

    Returns:
    - A list of filenames in the collection.
    """
    try:
        # Call the MilvusManager method to list all files
        filenames = milvus_manager.list_files(tenant_id=uuid)
        if not filenames:
            return {"message": "No files found in the collection."}
        
        logging.info(f"Files listed successfully for UUID {uuid}.")
        return {"filenames": filenames}

    except Exception as e:
        logging.error(f"Error occurred while listing files: {e}")
        raise HTTPException(status_code=500, detail=f"Error occurred while listing files: {e}")

@app.delete("/delete-file/")
async def delete_file(uuid: str = Header(...), filename: str = Header(...)):
    """
    Deletes a file from the Milvus collection based on the filename.

    Args:
    - uuid: The tenant UUID for identifying the collection.
    - filename: The name of the file to be deleted.

    Returns:
    - Success message indicating the number of deleted entities or an error message.
    """
    try:
        # Call the MilvusManager method to delete the file by filename
        delete_message = milvus_manager.delete_file_by_filename(tenant_id=uuid, filename=filename)
        logging.info(f"File '{filename}' deleted successfully for UUID {uuid}.")
        return {"message": delete_message}

    except Exception as e:
        logging.error(f"Error occurred while deleting file '{filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Error occurred while deleting file: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app", host="0.0.0.0", port=8000, reload=True)
