"""
Claude Baseline Implementation
Simple Claude implementation without hallucination mitigation techniques
"""

import os
from typing import Dict, List, Optional
from anthropic import Anthropic
import logging

logger = logging.getLogger(__name__)


class ClaudeBaseline:
    """Claude baseline model for comparative evaluation"""
    
    def __init__(self, config: Dict):
        """Initialize Claude baseline
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model_name = config.get("model_name", "claude-3-sonnet-20240229")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 1024)
        
    def generate_response(self, query: str, context: Optional[str] = None) -> Dict:
        """Generate response using standard Claude
        
        Args:
            query: User query
            context: Optional context (not used in baseline)
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Simple prompt without any mitigation techniques
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a helpful AI assistant.",
                messages=[
                    {"role": "user", "content": query}
                ]
            )
            
            # Extract text content from response
            response_text = response.content[0].text if response.content else ""
            
            return {
                "response": response_text,
                "model": self.model_name,
                "confidence": 1.0,  # Baseline has no confidence estimation
                "escalated": False,
                "metadata": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "stop_reason": response.stop_reason
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating Claude response: {e}")
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