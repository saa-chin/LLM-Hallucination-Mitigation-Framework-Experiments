"""
Retrieval-Augmented Generation (RAG) Layer Implementation
Implements document retrieval and context grounding
"""

import os
from typing import Dict, List, Optional, Tuple
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RAGLayer:
    """Implements RAG for grounding responses in verified documents"""
    
    def __init__(self, config: Dict):
        """Initialize RAG layer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.chunk_size = config.get("chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 100)
        self.top_k = config.get("top_k", 5)
        self.similarity_threshold = config.get("similarity_threshold", 0.75)
        
        # Initialize embedding function
        embedding_model = config.get("embedding_model", "text-embedding-ada-002")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=embedding_model
        )
        
        # Initialize ChromaDB
        self.client = chromadb.Client()
        self.collection = None
        
    def initialize_knowledge_base(self, documents: List[Dict]):
        """Initialize knowledge base with documents
        
        Args:
            documents: List of documents with 'content' and 'metadata' fields
        """
        # Create or get collection
        self.collection = self.client.create_collection(
            name="financial_knowledge_base",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Process and add documents
        for doc in documents:
            chunks = self._chunk_document(doc["content"])
            for i, chunk in enumerate(chunks):
                self.collection.add(
                    documents=[chunk],
                    metadatas=[{
                        **doc.get("metadata", {}),
                        "chunk_index": i,
                        "source": doc.get("source", "unknown")
                    }],
                    ids=[f"{doc.get('id', 'doc')}_{i}"]
                )
        
        logger.info(f"Initialized knowledge base with {len(documents)} documents")
    
    def retrieve_context(self, query: str) -> Tuple[str, List[Dict]]:
        """Retrieve relevant context for query
        
        Args:
            query: User query
            
        Returns:
            Tuple of (concatenated context, list of retrieved chunks with metadata)
        """
        if not self.collection:
            logger.warning("Knowledge base not initialized")
            return "", []
        
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=self.top_k
        )
        
        # Extract and filter results by similarity threshold
        retrieved_chunks = []
        if results["documents"] and results["distances"]:
            documents = results["documents"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            
            for doc, dist, meta in zip(documents, distances, metadatas):
                # Convert distance to similarity (cosine similarity)
                similarity = 1 - dist
                if similarity >= self.similarity_threshold:
                    retrieved_chunks.append({
                        "content": doc,
                        "similarity": float(similarity),
                        "metadata": meta
                    })
        
        # Sort by similarity
        retrieved_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Concatenate context
        context_parts = []
        for chunk in retrieved_chunks:
            source = chunk["metadata"].get("source", "Document")
            context_parts.append(f"[Source: {source}]\n{chunk['content']}")
        
        context = "\n\n".join(context_parts)
        
        return context, retrieved_chunks
    
    def calculate_confidence(self, retrieved_chunks: List[Dict]) -> float:
        """Calculate confidence score based on retrieval quality
        
        Args:
            retrieved_chunks: List of retrieved chunks with similarity scores
            
        Returns:
            Confidence score between 0 and 1
        """
        if not retrieved_chunks:
            return 0.0
        
        # Calculate weighted average of similarity scores
        similarities = [chunk["similarity"] for chunk in retrieved_chunks]
        
        # Weight earlier results more heavily
        weights = [1.0 / (i + 1) for i in range(len(similarities))]
        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
        weight_total = sum(weights)
        
        confidence = weighted_sum / weight_total if weight_total > 0 else 0.0
        
        # Apply threshold penalty
        if max(similarities) < self.similarity_threshold:
            confidence *= 0.5
        
        return float(confidence)
    
    def _chunk_document(self, content: str) -> List[str]:
        """Chunk document into overlapping segments
        
        Args:
            content: Document content
            
        Returns:
            List of text chunks
        """
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words)
            if chunk:
                chunks.append(chunk)
        
        return chunks