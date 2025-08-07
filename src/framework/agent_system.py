"""
Multi-Layered Agent System Implementation
Combines prompt engineering, RAG, and confidence-based escalation
"""

import os
from typing import Dict, List, Optional
from openai import OpenAI
import logging

from .prompt_layer import PromptEngineeringLayer
from .rag_layer import RAGLayer

logger = logging.getLogger(__name__)


class MultiLayeredAgent:
    """Agent system implementing the full mitigation framework"""
    
    def __init__(self, config: Dict):
        """Initialize multi-layered agent
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Initialize layers
        self.prompt_layer = PromptEngineeringLayer(config["prompt_engineering"])
        self.rag_layer = RAGLayer(config["rag"])
        
        # Agent configuration
        self.confidence_thresholds = config["agent"]["confidence_thresholds"]
        self.escalation_enabled = config["agent"]["escalation_enabled"]
        
        # Model configuration
        self.model_name = config.get("base_model", "gpt-4-turbo-preview")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 1024)
    
    def initialize_knowledge_base(self, documents: List[Dict]):
        """Initialize RAG knowledge base
        
        Args:
            documents: List of documents for knowledge base
        """
        self.rag_layer.initialize_knowledge_base(documents)
    
    def generate_response(self, query: str) -> Dict:
        """Generate response using multi-layered approach
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Step 1: Classify query domain
            domain = self._classify_domain(query)
            threshold = self.confidence_thresholds.get(domain, 0.75)
            
            # Step 2: Retrieve relevant context
            context, retrieved_chunks = self.rag_layer.retrieve_context(query)
            
            # Step 3: Calculate retrieval confidence
            retrieval_confidence = self.rag_layer.calculate_confidence(retrieved_chunks)
            
            # Step 4: Check if we should escalate
            if self.escalation_enabled and retrieval_confidence < threshold:
                return self._generate_escalation_response(
                    query, domain, retrieval_confidence, threshold
                )
            
            # Step 5: Construct prompt with engineering techniques
            prompt = self.prompt_layer.construct_prompt(query, context)
            
            # Step 6: Generate response
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            response_text = response.choices[0].message.content
            
            # Step 7: Extract reasoning if CoT was used
            reasoning_data = self.prompt_layer.extract_reasoning(response_text)
            
            return {
                "response": reasoning_data["answer"],
                "model": f"{self.model_name}_multilayer",
                "confidence": float(retrieval_confidence),
                "escalated": False,
                "domain": domain,
                "metadata": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "retrieved_chunks": len(retrieved_chunks),
                    "reasoning": reasoning_data["reasoning"],
                    "context_used": bool(context),
                    "techniques_applied": [
                        "prompt_engineering",
                        "rag" if context else None,
                        "cot" if reasoning_data["reasoning"] else None
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating multi-layered response: {e}")
            return {
                "response": None,
                "error": str(e),
                "model": f"{self.model_name}_multilayer",
                "confidence": 0.0,
                "escalated": False
            }
    
    def _classify_domain(self, query: str) -> str:
        """Classify query domain for threshold selection
        
        Args:
            query: User query
            
        Returns:
            Domain classification
        """
        query_lower = query.lower()
        
        # Compliance keywords
        compliance_keywords = ["regulation", "compliance", "legal", "law", "rule", "policy"]
        if any(keyword in query_lower for keyword in compliance_keywords):
            return "compliance"
        
        # Financial keywords
        financial_keywords = ["fee", "rate", "fund", "investment", "portfolio", "account"]
        if any(keyword in query_lower for keyword in financial_keywords):
            return "financial"
        
        return "general"
    
    def _generate_escalation_response(self, query: str, domain: str, 
                                    confidence: float, threshold: float) -> Dict:
        """Generate escalation response when confidence is too low
        
        Args:
            query: User query
            domain: Query domain
            confidence: Retrieval confidence
            threshold: Required threshold
            
        Returns:
            Escalation response dictionary
        """
        escalation_message = (
            "I don't have sufficient information to provide a complete answer to your question. "
            "This query requires access to specific documentation that I couldn't locate with "
            "high confidence. Please contact a human representative who can assist you with "
            "the most current and accurate information."
        )
        
        return {
            "response": escalation_message,
            "model": f"{self.model_name}_multilayer",
            "confidence": float(confidence),
            "escalated": True,
            "domain": domain,
            "metadata": {
                "escalation_reason": "low_confidence",
                "confidence_score": float(confidence),
                "required_threshold": float(threshold),
                "techniques_applied": ["prompt_engineering", "rag", "escalation"]
            }
        }
    
    def batch_generate(self, queries: List[str]) -> List[Dict]:
        """Generate responses for multiple queries
        
        Args:
            queries: List of user queries
            
        Returns:
            List of response dictionaries
        """
        responses = []
        for query in queries:
            response = self.generate_response(query)
            responses.append(response)
        return responses