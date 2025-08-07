"""
SelfCheckGPT Implementation for Hallucination Detection
Based on: https://arxiv.org/abs/2303.08896
"""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class SelfCheckGPT:
    """SelfCheckGPT implementation for hallucination detection"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize SelfCheckGPT
        
        Args:
            embedding_model: Name of sentence transformer model for embeddings
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        
    def check_consistency(self, 
                         original_response: str, 
                         sampled_responses: List[str]) -> Dict:
        """Check consistency between original and sampled responses
        
        Args:
            original_response: Original model response
            sampled_responses: List of sampled responses for the same query
            
        Returns:
            Dictionary containing consistency scores and hallucination likelihood
        """
        # Split responses into sentences
        original_sentences = self._split_into_sentences(original_response)
        
        if not original_sentences:
            return {
                "consistency_score": 1.0,
                "hallucination_score": 0.0,
                "sentence_scores": []
            }
        
        # Calculate consistency for each sentence
        sentence_scores = []
        
        for orig_sent in original_sentences:
            # Get embeddings for original sentence
            orig_embedding = self.embedding_model.encode([orig_sent])
            
            # Calculate similarity with sentences from sampled responses
            similarities = []
            for sampled_resp in sampled_responses:
                sampled_sentences = self._split_into_sentences(sampled_resp)
                if sampled_sentences:
                    sampled_embeddings = self.embedding_model.encode(sampled_sentences)
                    # Get max similarity for this sentence
                    sim_scores = cosine_similarity(orig_embedding, sampled_embeddings)[0]
                    max_sim = np.max(sim_scores)
                    similarities.append(max_sim)
            
            # Average similarity across all sampled responses
            avg_similarity = np.mean(similarities) if similarities else 0.0
            sentence_scores.append({
                "sentence": orig_sent,
                "consistency_score": float(avg_similarity),
                "hallucination_likelihood": float(1.0 - avg_similarity)
            })
        
        # Calculate overall scores
        consistency_scores = [s["consistency_score"] for s in sentence_scores]
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        overall_hallucination = 1.0 - overall_consistency
        
        return {
            "consistency_score": float(overall_consistency),
            "hallucination_score": float(overall_hallucination),
            "sentence_scores": sentence_scores,
            "flagged_sentences": [
                s for s in sentence_scores 
                if s["hallucination_likelihood"] > 0.5
            ]
        }
    
    def batch_check(self, 
                   original_responses: List[str], 
                   sampled_responses_list: List[List[str]]) -> List[Dict]:
        """Check consistency for multiple responses
        
        Args:
            original_responses: List of original responses
            sampled_responses_list: List of sampled response lists
            
        Returns:
            List of consistency check results
        """
        results = []
        for orig, sampled in zip(original_responses, sampled_responses_list):
            result = self.check_consistency(orig, sampled)
            results.append(result)
        return results
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLTK or spaCy)
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def get_hallucination_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate hallucination metrics
        
        Args:
            results: List of consistency check results
            
        Returns:
            Dictionary containing aggregate metrics
        """
        hallucination_scores = [r["hallucination_score"] for r in results]
        flagged_count = sum(1 for r in results if r["hallucination_score"] > 0.5)
        
        return {
            "mean_hallucination_score": float(np.mean(hallucination_scores)),
            "std_hallucination_score": float(np.std(hallucination_scores)),
            "hallucination_rate": float(flagged_count / len(results)) if results else 0.0,
            "total_evaluated": len(results),
            "total_flagged": flagged_count
        }