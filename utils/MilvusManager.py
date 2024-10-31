import logging
import uuid
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility
)


class MilvusManager:
    def __init__(self, host="127.0.0.1", port="19530"):
        # Connect to Milvus server
        connections.connect(host=host, port=port, db_name="my_database")
    
    def _sanitize_tenant_id(self, tenant_id):
        """Helper function to sanitize tenant_id by replacing hyphens with underscores."""
        return tenant_id.replace("-", "_")
    def has_collection(self, collection_name):
        if utility.has_collection(collection_name):
            print(f"Collection {collection_name} already exists.")
            return

    def create_tenant_collection(self, tenant_id, fields):
        """
        Creates a collection for a tenant identified by tenant_id (UUID).
        Accepts a list of FieldSchema objects for the schema.

        :param tenant_id: Unique identifier for the tenant (UUID).
        :param fields: List of FieldSchema objects defining the schema.
        """
        try:
            # Sanitize tenant ID to avoid any issues with special characters
            sanitized_tenant_id = self._sanitize_tenant_id(tenant_id)
            collection_name = f"tenant_{sanitized_tenant_id}"

            # Check if the collection already exists
            if self.has_collection(collection_name):
                return "Collection already exists. Try again!"

            # Create collection schema dynamically based on fields
            schema = CollectionSchema(fields, description=f"Collection for tenant {sanitized_tenant_id}")

            # Create the collection
            collection = Collection(name=collection_name, schema=schema)
            self.create_index(collection, field_name="vector")

            logging.info(f"Collection {collection_name} created.")
            return tenant_id

        except Exception as e:
            # Log the error and raise an exception with more information
            logging.error(f"Error occurred while creating collection : {e}")
            raise Exception(f"Failed to create collection. Error: {e}")


    def filename_exists(self, tenant_id, filename):
            """
            Check if a file with the given filename already exists in the tenant's collection.
            
            :param tenant_id: Unique identifier for the tenant (UUID).
            :param filename: The filename to check.
            :return: True if the filename exists, False otherwise.
            """
            sanitized_tenant_id = self._sanitize_tenant_id(tenant_id)
            collection_name = f"tenant_{sanitized_tenant_id}"

            if not utility.has_collection(collection_name):
                raise Exception(f"Collection {collection_name} does not exist.")

            collection = Collection(collection_name)
            collection.load()

            try:
                # Query to check if the filename already exists
                query_result = collection.query(expr=f'filename == "{filename}"', output_fields=["filename"])
                return len(query_result) > 0

            except Exception as e:
                logging.error(f"Failed to check if filename '{filename}' exists: {e}")
                raise Exception(f"Error checking filename: {e}")

    # def insert_data(self, tenant_id, data):
    #     """
    #     Insert data into the tenant-specific collection and create an index.
        
    #     :param tenant_id: Unique identifier for the tenant (UUID).
    #     :param data: List of data records to insert (should match the schema of the collection).
    #     """
    #     sanitized_tenant_id = self._sanitize_tenant_id(tenant_id)
    #     collection_name = f"tenant_{sanitized_tenant_id}"

    #     if not utility.has_collection(collection_name):
    #         raise Exception(f"Collection {collection_name} does not exist.")

    #     # Check if the filename already exists in the collection
    #     for record in data:
    #         if 'filename' in record and self.filename_exists(tenant_id, record['filename']):
    #             raise Exception(f"Filename '{record['filename']}' already exists. Insertion aborted.")

    #     collection = Collection(collection_name)
        
    #     # Insert data
    #     collection.insert(data)
    #     collection.flush()
    #     print(f"Data inserted into collection {collection_name}.")

    #     # Create an index for the vector field (if needed)
    #     self.create_index(collection, field_name="vector")
   
    def insert_data(self, tenant_id, data):
        """
        Insert data into the tenant-specific collection and create an index.
        
        :param tenant_id: Unique identifier for the tenant (UUID).
        :param data: List of data records to insert (should match the schema of the collection).
        """
        sanitized_tenant_id = self._sanitize_tenant_id(tenant_id)
        collection_name = f"tenant_{sanitized_tenant_id}"

        if not utility.has_collection(collection_name):
            raise Exception(f"Collection {collection_name} does not exist.")

        collection = Collection(collection_name)
        
        # Insert data
        collection.insert(data)
        collection.flush()
        print(f"Data inserted into collection {collection_name}.")

        # Create an index for the vector field (if needed)
        self.create_index(collection, field_name="vector")

    def create_index(self, collection, field_name="vector", index_params=None):
        """
        Create an index for a field in a collection. Default is for vector fields.
        
        :param collection: The Milvus collection instance.
        :param field_name: Name of the field to index.
        :param index_params: Optional parameters for index creation (type, metric, etc.).
        """
        if index_params is None:
            index_params = {
                "index_type": "IVF_FLAT",  # Default index type, can be customized
                "metric_type": "L2",       # Default metric type (Euclidean distance)
                "params": {"nlist": 128}   # Default index params
            }

        # Create an index on the specified field
        collection.create_index(field_name=field_name, index_params=index_params)
        print(f"Index created for field {field_name} in collection {collection.name}.")

    def search(self, tenant_id, query_vectors, top_k=5, search_params=None):
        """
        Search for the most similar vectors in a tenant-specific collection.
        
        :param tenant_id: Unique identifier for the tenant (UUID).
        :param query_vectors: List of query vectors.
        :param top_k: Number of top results to return.
        :param search_params: Search parameters (depends on index type).
        :return: Search results including text and filename.
        """
        sanitized_tenant_id = self._sanitize_tenant_id(tenant_id)
        collection_name = f"tenant_{sanitized_tenant_id}"

        if not utility.has_collection(collection_name):
            raise Exception(f"Collection {collection_name} does not exist.")

        collection = Collection(collection_name)
        collection.load()

        if search_params is None:
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}  # Default search params

        try:
            # Perform the search
            results = collection.search(
                data=query_vectors,
                anns_field="vector",  # The field we indexed
                param=search_params,
                limit=top_k,
                output_fields=["text", "filename"]  # Specify the fields to return
            )

            # Format the results
            formatted_results = []
            for result in results:
                for hit in result:
                    formatted_results.append({
                        "id": hit.id,
                        "distance": hit.distance,
                        "text": hit.entity.get("text"),
                        "filename": hit.entity.get("filename")
                    })

            return formatted_results

        except Exception as e:
            logging.error(f"Search failed: {e}")
            raise

    def list_collections(self):
        """
        List all collections available on the server.
        """
        collections = utility.list_collections()
        return collections


    def delete_file_by_filename(self, tenant_id, filename):
        """
        Delete entities from a tenant's collection where the 'filename' matches the provided filename.
        
        :param tenant_id: Unique identifier for the tenant (UUID).
        :param filename: The filename to match and delete.
        :return: Number of deleted entities or a message indicating success/failure.
        """
        sanitized_tenant_id = self._sanitize_tenant_id(tenant_id)
        collection_name = f"tenant_{sanitized_tenant_id}"

        if not utility.has_collection(collection_name):
            raise Exception(f"Collection {collection_name} does not exist.")

        collection = Collection(collection_name)
        collection.load()

        try:
            # Delete the entities where the 'filename' matches the given filename
            delete_expression = f'filename == "{filename}"'
            delete_result = collection.delete(expr=delete_expression)

            # Return the number of deleted entities or an appropriate message
            num_deleted = delete_result.delete_count
            if num_deleted > 0:
                return f"{num_deleted} entities with filename '{filename}' were deleted."
            else:
                return f"No entities found with filename '{filename}'."

        except Exception as e:
            logging.error(f"Failed to delete file with filename '{filename}' from collection: {e}")
            raise Exception(f"Failed to delete file. Error: {e}")

    def search_with_filter(self, tenant_id, query_vectors, top_k=5, search_params=None, filter_expr=None):
        """
        Search for the most similar vectors in a tenant-specific collection, with an optional filter expression.

        :param tenant_id: Unique identifier for the tenant (UUID).
        :param query_vectors: List of query vectors.
        :param top_k: Number of top results to return.
        :param search_params: Search parameters (depends on index type).
        :param filter_expr: Optional expression to filter the search by filenames or other criteria.
        :return: Search results including text and filename.
        """
        sanitized_tenant_id = self._sanitize_tenant_id(tenant_id)
        collection_name = f"tenant_{sanitized_tenant_id}"

        if not utility.has_collection(collection_name):
            raise Exception(f"Collection {collection_name} does not exist.")

        collection = Collection(collection_name)
        collection.load()

        if search_params is None:
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}  # Default search params

        try:
            # Perform the search with an optional filter
            if filter_expr:
                results = collection.search(
                    data=query_vectors,
                    anns_field="vector",
                    param=search_params,
                    limit=top_k,
                    expr=filter_expr,  # Apply the filter expression here
                    output_fields=["text", "filename"]
                )
            else:
                results = collection.search(
                    data=query_vectors,
                    anns_field="vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=["text", "filename"]
                )

            # Format the results
            formatted_results = []
            for result in results:
                for hit in result:
                    formatted_results.append({
                        "id": hit.id,
                        "distance": hit.distance,
                        "text": hit.entity.get("text"),
                        "filename": hit.entity.get("filename")
                    })

            return formatted_results

        except Exception as e:
            logging.error(f"Search with filter failed: {e}")
            raise

    def list_files(self, tenant_id, limit=1000):
        """
        List all unique files in a tenant-specific collection by retrieving the 'filename' field.

        :param tenant_id: Unique identifier for the tenant (UUID).
        :param limit: Limit on the number of files to retrieve (default: 1000).
        :return: A list of unique filenames in the collection.
        """
        try:
            sanitized_tenant_id = self._sanitize_tenant_id(tenant_id)
            collection_name = f"tenant_{sanitized_tenant_id}"

            if not utility.has_collection(collection_name):
                raise Exception(f"Collection {collection_name} does not exist.")

            collection = Collection(collection_name)
            collection.load()

            # Use a limit to retrieve filenames (since we can't use an empty expression without limit)
            results = collection.query(expr="", output_fields=["filename"], limit=limit)

            # Extract filenames and remove duplicates by converting to a set
            filenames = list(set(result['filename'] for result in results if 'filename' in result))

            return filenames

        except Exception as e:
            logging.error(f"Error while listing files: {e}")
            raise Exception(f"Error while listing files: {e}")
