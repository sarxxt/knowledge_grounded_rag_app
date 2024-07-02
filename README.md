
# LLM RAG

Generate the Query Response From Specific pdf using OpenAI's LLM.



## Features

- Can Uplaod a PDF
- Query to LLM from PDF
- Clean the Vector Stores


## API Reference

#### Post the PDF to create a vector Store against the UUID
#### Post Query to get response from the PDF data
#### Delete the Vector Stores from the memory against the UUID

```http
  POST /uploadpdf/
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `UUID` | `Header` | **Required**. UUID of the User|
| `File` | `File` | **Required**. PDF file  |



#### Post the query and relative uuid in header

```http
  POST /query/
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `UUID` | `Header` | **Required**. UUID of the User|
| `query`      | `String` | **Required**. Query from PDF|


#### Post uuid in header

```http
  POST /clean_db/
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `UUID` | `Header` | **Required**. UUID of the User|



## Environment Variables

To run this project, you will need to add the following environment variable to your .env file

`openai_api_key`



## Run Locally

#### Backend Setup

Clone the project

```bash
 https://github.com/rizwansaleem01/integrity_chatbot
```

Go to the project directory

```bash
  cd integrity_chatbot
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  uvicorn main:app --host 0.0.0.0 --port 8000
```
