"""
Prompt Engineering Layer Implementation
Implements few-shot prompting, role-playing, and chain-of-thought reasoning
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PromptEngineeringLayer:
    """Implements prompt engineering techniques for hallucination mitigation"""
    
    def __init__(self, config: Dict):
        """Initialize prompt engineering layer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.few_shot_examples = config.get("few_shot_examples", 3)
        self.role_prompt = config.get("role_prompt", 
            "You are a helpful AI assistant focused on providing accurate, factual information.")
        self.cot_enabled = config.get("cot_enabled", True)
        
        # Financial domain examples for few-shot learning
        self.financial_examples = [
            {
                "query": "What are the fees for the Quantum Fund?",
                "response": "I need to check the specific documentation for the Quantum Fund fees. Without access to the current fee schedule, I cannot provide accurate fee information."
            },
            {
                "query": "Is ACME Corp a good investment?",
                "response": "I cannot provide investment advice or recommendations. Investment decisions should be based on your individual financial situation, risk tolerance, and consultation with qualified financial advisors."
            },
            {
                "query": "What is the current interest rate for savings accounts?",
                "response": "Interest rates vary by institution and account type. I would need access to specific bank documentation to provide accurate current rates. Please check with your financial institution for their current rates."
            }
        ]
    
    def construct_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Construct prompt with engineering techniques
        
        Args:
            query: User query
            context: Optional context from RAG
            
        Returns:
            Engineered prompt
        """
        prompt_parts = []
        
        # Add role prompt
        prompt_parts.append(f"System: {self.role_prompt}")
        prompt_parts.append("Important: Only provide information that you are certain about. If uncertain, acknowledge the limitation.")
        
        # Add few-shot examples if applicable
        if self._is_financial_query(query):
            prompt_parts.append("\nHere are some examples of appropriate responses:")
            for i, example in enumerate(self.financial_examples[:self.few_shot_examples]):
                prompt_parts.append(f"\nExample {i+1}:")
                prompt_parts.append(f"User: {example['query']}")
                prompt_parts.append(f"Assistant: {example['response']}")
        
        # Add context if provided
        if context:
            prompt_parts.append(f"\nContext: {context}")
            prompt_parts.append("Use only the information provided in the context above.")
        
        # Add chain-of-thought instruction if enabled
        if self.cot_enabled:
            prompt_parts.append("\nBefore answering, think step-by-step about:")
            prompt_parts.append("1. What information is being requested")
            prompt_parts.append("2. What verified information is available")
            prompt_parts.append("3. Whether you can provide a complete and accurate answer")
        
        # Add the actual query
        prompt_parts.append(f"\nUser Query: {query}")
        prompt_parts.append("\nAssistant:")
        
        return "\n".join(prompt_parts)
    
    def _is_financial_query(self, query: str) -> bool:
        """Check if query is related to financial domain
        
        Args:
            query: User query
            
        Returns:
            Boolean indicating if query is financial
        """
        financial_keywords = [
            "fee", "investment", "fund", "rate", "return", "portfolio",
            "stock", "bond", "dividend", "interest", "account", "loan",
            "mortgage", "credit", "finance", "money", "dollar", "cost"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in financial_keywords)
    
    def extract_reasoning(self, response: str) -> Dict:
        """Extract reasoning steps from CoT response
        
        Args:
            response: Model response
            
        Returns:
            Dictionary with reasoning and final answer
        """
        # Simple extraction - can be improved with more sophisticated parsing
        if "step-by-step" in response.lower() or "step 1:" in response.lower():
            parts = response.split("\n\n")
            if len(parts) > 1:
                return {
                    "reasoning": parts[0],
                    "answer": "\n\n".join(parts[1:])
                }
        
        return {
            "reasoning": "",
            "answer": response
        }