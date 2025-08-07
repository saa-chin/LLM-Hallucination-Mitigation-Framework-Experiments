"""
GPT-4 Baseline Implementation
Simple GPT-4 implementation without hallucination mitigation techniques
"""

import os
from typing import Dict, List, Optional
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class GPT4Baseline:
    """GPT-4 baseline model for comparative evaluation"""
    
    def __init__(self, config: Dict):
        """Initialize GPT-4 baseline
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = config.get("model_name", "gpt-4-turbo-preview")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 1024)
        
    def generate_response(self, query: str, context: Optional[str] = None) -> Dict:
        """Generate response using standard GPT-4
        
        Args:
            query: User query
            context: Optional context (not used in baseline)
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Simple prompt without any mitigation techniques
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": query}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return {
                "response": response.choices[0].message.content,
                "model": self.model_name,
                "confidence": 1.0,  # Baseline has no confidence estimation
                "escalated": False,
                "metadata": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating GPT-4 response: {e}")
            return {
                "response": None,
                "error": str(e),
                "model": self.model_name,
                "confidence": 0.0,
                "escalated": False
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