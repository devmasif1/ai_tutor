import os
import zipfile
import tempfile
import shutil
import re
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from datetime import datetime
import uuid

from src.document_processor import DocumentProcessor
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch


class MongoKnowledgeBase:
    """MongoDB Atlas-based Knowledge Base with per-user collections"""
    
    def __init__(
            self, 
            user_id: str,
            username: str = None,
            mongo_uri: str = None,       
        ):

        self.user_id = user_id
        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI")  
        self.username = username or user_id
        
        if not self.mongo_uri:
            raise ValueError("MongoDB URI not provided. Set MONGODB_URI environment variable.")
        
        # Initialize MongoDB client
        self.mongo_client = MongoClient(self.mongo_uri)
        
        # Database setup
        self.db_name = "ai_tutor_db"
        
        # Create user-specific collection name (sanitize for MongoDB compatibility)
        self.collection_name = self._generate_user_collection_name(username)
        self.index_name = f"vector_index_{self._sanitize_name(user_id)}"
        
        self.collection = self.mongo_client[self.db_name][self.collection_name]
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.embedding_function = OpenAIEmbeddings()
        
        # Initialize MongoDB Atlas Vector Search with user-specific collection
        self.vectorstore = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embedding_function,
            index_name=self.index_name,
            text_key="content",
            embedding_key="embedding"
        )
        
        # Try to create vector search index if needed (non-blocking)
        self._ensure_vector_index()
        
        print(f"[INFO] MongoKnowledgeBase initialized for user: {self.username}")
        print(f"[INFO] User Collection: {self.collection_name}")
        print(f"[INFO] Index: {self.index_name}")
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for MongoDB collection naming rules"""
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        # Ensure it starts with letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = f"user_{sanitized}"
        # Limit length (MongoDB collection names have a 127 byte limit)
        return sanitized[:50]
    
    def _generate_user_collection_name(self,username: str = None) -> str:
        """Generate a unique collection name for the user"""
        base_name = username 
        sanitized_name = self._sanitize_name(base_name)
        

        collection_name = f"vectors_db_{sanitized_name}"

        return collection_name
    
    def _ensure_vector_index(self):
        """Ensure vector search index exists for this user's collection"""
        try:
            # Check if index already exists
            existing_indexes = list(self.collection.list_search_indexes())
            index_exists = any(idx.get('name') == self.index_name for idx in existing_indexes)
            
            if not index_exists:
                print(f"[INFO] Creating vector search index: {self.index_name}")
                
                # Create the vector search index
                index_definition = {
                    "name": self.index_name,
                    "definition": {
                        "mappings": {
                            "dynamic": True,
                            "fields": {
                                "embedding": {
                                    "type": "knnVector",
                                    "dimensions": 1536,  # OpenAI embedding dimensions
                                    "similarity": "cosine"
                                },
                                "content": {
                                    "type": "string"
                                }
                            }
                        }
                    }
                }
                
                self.collection.create_search_index(index_definition)
                print(f"[INFO] Vector search index creation initiated: {self.index_name}")
                print("[INFO] Note: Index creation may take a few minutes to complete")
            else:
                print(f"[INFO] Vector search index already exists: {self.index_name}")
                
        except Exception as e:
            print(f"[WARNING] Could not ensure vector index: {e}")
            print("[INFO] You may need to create the vector search index manually in MongoDB Atlas")

    def get_directory_summary(self, path: str) -> str:
        """Generate a formatted directory structure summary"""
        if not os.path.exists(path):
            return f"Path does not exist: {path}"
        
        summary_lines = []
        
        if os.path.isfile(path):
            file_size = os.path.getsize(path)
            return f"File: {os.path.basename(path)} ({file_size:,} bytes)"
        
        total_files = 0
        total_size = 0
        supported_extensions = set(self.doc_processor.get_supported_formats())
        
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = "  " * level
            folder_name = os.path.basename(root) if root != path else os.path.basename(path)
            summary_lines.append(f"{indent}{folder_name}/")
            
            sub_indent = "  " * (level + 1)
            for d in sorted(dirs):
                summary_lines.append(f"{sub_indent}{d}/")
            
            supported_files = []
            other_files = []
            
            for f in sorted(files):
                file_path = os.path.join(root, f)
                file_ext = os.path.splitext(f)[1].lower()
                file_size = os.path.getsize(file_path)
                total_files += 1
                total_size += file_size
                
                file_info = f"{f} ({file_size:,} bytes)"
                if file_ext in supported_extensions:
                    supported_files.append(f"{sub_indent}✓ {file_info}")
                else:
                    other_files.append(f"{sub_indent}✗ {file_info}")
            
            summary_lines.extend(supported_files)
            summary_lines.extend(other_files)
        
        summary = "\n".join(summary_lines)
        summary += f"\n\nSummary: {total_files} files, {total_size:,} bytes total"
        summary += f"\n✓ = Supported format, ✗ = Unsupported format"
        
        return summary
    
    def add_content(self, path: str) -> str:
        """Add content from file, folder, or ZIP to the user's knowledge base"""
        if not os.path.exists(path):
            return f"Error: Path does not exist: {path}"
        
        summary = self.get_directory_summary(path)
        result_messages = [f"Directory/File Structure:\n{summary}\n"]
        
        try:
            documents = []
            temp_dir = None
            
            # Handle different path types
            if path.lower().endswith('.zip'):
                result_messages.append("Processing ZIP archive...")
                temp_dir = tempfile.mkdtemp()
                
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                documents = self.doc_processor.load_directory(temp_dir)
                
                for doc in documents:
                    doc.metadata['original_source'] = path
                    doc.metadata['extracted_from_zip'] = True
            
            elif os.path.isdir(path):
                result_messages.append("Processing directory...")
                documents = self.doc_processor.load_directory(path)
            
            elif os.path.isfile(path):
                result_messages.append("Processing file...")
                documents = self.doc_processor.load_any_file(path)
            
            else:
                return "Error: Invalid path type."
            
            if not documents:
                result_messages.append("No supported documents found or all documents were empty.")
                return "\n".join(result_messages)
            
            # Debug: Check original doc_types
            print(f"[DEBUG] Original documents loaded: {len(documents)}")
            original_doc_types = [doc.metadata.get('doc_type', 'MISSING') for doc in documents[:3]]
            print(f"[DEBUG] Sample original doc_types: {original_doc_types}")
            
            # Split documents into chunks
            chunks = self.doc_processor.split_documents(documents)
            if not chunks:
                result_messages.append("No valid content chunks could be created.")
                return "\n".join(result_messages)
            
            # Debug: Check chunk doc_types after splitting
            print(f"[DEBUG] Chunks created: {len(chunks)}")
            chunk_doc_types = [chunk.metadata.get('doc_type', 'MISSING') for chunk in chunks[:3]]
            print(f"[DEBUG] Sample chunk doc_types: {chunk_doc_types}")
            
            processed_chunks = []
            for chunk in chunks:
                # Get doc_type with better fallback logic
                doc_type = chunk.metadata.get('doc_type')
                
                # If doc_type is missing or empty, try to infer from source
                if not doc_type or doc_type == 'unknown' or doc_type == 'MISSING':
                    source_path = chunk.metadata.get('source', '')
                    if source_path:
                        # Extract filename without extension as doc_type
                        doc_type = os.path.splitext(os.path.basename(source_path))[0]
                        print(f"[DEBUG] Inferred doc_type '{doc_type}' from source: {source_path}")
                    else:
                        doc_type = 'unknown_document'
                
                processed_doc = Document(
                    page_content=chunk.page_content,
                    metadata={
                        "doc_type": doc_type,  
                        "source": chunk.metadata.get('source', 'unknown'),
                        "added_timestamp": datetime.now().isoformat(),
                        "page": chunk.metadata.get('page', 0),  

                    }
                )
                processed_chunks.append(processed_doc)
            
        
            final_doc_types = [doc.metadata.get('doc_type') for doc in processed_chunks[:5]]
            print(f"[DEBUG] Final doc_types (first 5): {final_doc_types}")
            unique_doc_types = set(doc.metadata.get('doc_type') for doc in processed_chunks)
            print(f"[DEBUG] All unique doc_types: {unique_doc_types}")
            
            print(f"[DEBUG] Adding {len(processed_chunks)} chunks to collection: {self.collection_name}")
            
            # Add documents to vectorstore
            doc_ids = self.vectorstore.add_documents(processed_chunks)
            print(f"[DEBUG] Successfully added documents with IDs: {len(doc_ids)}")
            
            # Verify documents were added correctly
            verification_count = self.collection.count_documents({})
            print(f"[DEBUG] Verification: Collection {self.collection_name} now has {verification_count} documents")
            
            # Get document types from original documents for summary
            doc_types = set(doc.metadata.get('doc_type', 'unknown') for doc in processed_chunks)
            result_messages.append(
                f"✅ Added {len(processed_chunks)} chunks from {len(documents)} documents to user collection.\n"
                f"Collection: {self.collection_name}\n"
                f"Document types found: {', '.join(sorted(doc_types))}\n"
                f"Generated {len(doc_ids)} document IDs"
            )
            
            # Clean up
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # Add stats
            result_messages.append(f"\n{self.investigate_vectors()}")
            
            return "\n".join(result_messages)
            
        except Exception as e:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            print(f"[ERROR] Error in add_content: {str(e)}")


    def similarity_search_with_user_filter(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search - much simpler now with per-user collections"""
        try:
            print(f"[DEBUG] Performing similarity search in collection: {self.collection_name}")
            
            # Direct search - no filtering needed since each user has their own collection
            docs = self.vectorstore.similarity_search(query=query, k=k)
            
            print(f"[DEBUG] Found {len(docs)} documents in user's collection")
            return docs
            
        except Exception as e:
            print(f"[ERROR] Similarity search failed: {e}")
            return []
    
    def investigate_vectors(self) -> str:
        """Get information about the user's vector store"""
        try:
            total_docs = self.collection.count_documents({})
            
            if total_docs == 0:
                return f"Vector store is empty for user {self.user_id} (Collection: {self.collection_name})"
            
            # Get document types distribution
            doc_types = list(self.collection.aggregate([
                {"$group": {"_id": "$metadata.doc_type", "count": {"$sum": 1}}}
            ]))
            
            # Get sample document for dimension info
            sample = self.collection.find_one({"embedding": {"$exists": True}})
            dimensions = len(sample["embedding"]) if sample and "embedding" in sample else 0
            
            # Check vector search index status
            try:
                indexes = list(self.collection.list_search_indexes())
                vector_index_status = "not_found"
                for idx in indexes:
                    if idx.get('name') == self.index_name:
                        vector_index_status = idx.get('status', 'unknown')
                        break
            except:
                vector_index_status = "not_available"
            
            result = f"MongoDB Atlas Vector store for user {self.username}:\n"
            result += f"- Collection: {self.collection_name}\n"
            result += f"- Total documents: {total_docs:,}\n"
            result += f"- Embedding dimensions: {dimensions:,}\n"
            
            return result
            
        except Exception as e:
            return f"Error investigating vectors: {str(e)}"
    
    def get_vector_store(self) -> MongoDBAtlasVectorSearch:
        """Get the vector store instance"""
        return self.vectorstore
    
    def clear_knowledge_base(self) -> str:
        """Clear the user's entire collection"""
        try:
            # Drop the entire collection for this user
            self.collection.drop()
            print(f"[INFO] Dropped collection: {self.collection_name}")
            
            # Recreate the collection reference
            self.collection = self.mongo_client[self.db_name][self.collection_name]
            
            # Reinitialize vectorstore with the new collection
            self.vectorstore = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embedding_function,
                index_name=self.index_name,
                text_key="content",
                embedding_key="embedding"
            )
            
            # Recreate index
            self._ensure_vector_index()
            
            return f"Cleared all documents from user collection: {self.collection_name}"
            
        except Exception as e:
            return f"Error clearing knowledge base: {str(e)}"
    
    def get_user_retriever(self, search_kwargs=None):
        """Get a retriever for this user's collection"""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        def custom_retriever(query: str) -> List[Document]:
            k = search_kwargs.get("k", 5)
            return self.similarity_search_with_user_filter(query, k)
        
        return custom_retriever

    def get_user_document_count(self) -> int:
        """Get total document count for this user"""
        try:
            return self.collection.count_documents({})
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    def list_user_collections(self) -> List[str]:
        """List all user collections in the database (admin function)"""
        try:
            all_collections = self.mongo_client[self.db_name].list_collection_names()
            user_collections = [col for col in all_collections if col.startswith('vectors_')]
            return user_collections
        except Exception as e:
            print(f"Error listing user collections: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics (admin function)"""
        try:
            db = self.mongo_client[self.db_name]
            stats = db.command("dbStats")
            
            # Get user collection info
            user_collections = self.list_user_collections()
            collection_stats = {}
            
            for collection_name in user_collections:
                try:
                    col_stats = db.command("collStats", collection_name)
                    collection_stats[collection_name] = {
                        "documents": col_stats.get("count", 0),
                        "size_bytes": col_stats.get("size", 0),
                        "index_count": col_stats.get("nindexes", 0)
                    }
                except Exception as e:
                    collection_stats[collection_name] = {"error": str(e)}
            
            return {
                "database_stats": stats,
                "user_collections": collection_stats,
                "total_user_collections": len(user_collections)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def close(self):
        """Close MongoDB connection"""
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()
            print(f"[INFO] Closed MongoDB connection for user: {self.username}")