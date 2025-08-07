#!/usr/bin/env python3
"""
Test evaluation script with limited queries to verify setup
"""

import os
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv('.env')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.baselines.gpt4_baseline import GPT4Baseline
from src.baselines.claude_baseline import ClaudeBaseline
from src.framework.agent_system import MultiLayeredAgent
from src.evaluation.selfcheckgpt import SelfCheckGPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_evaluation():
    """Run a limited test evaluation"""
    
    # Load configuration
    with open('config/eval_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test queries - just a few for testing
    test_queries = {
        "general_knowledge": [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?"
        ],
        "financial_services": [
            "What are the fees for the Quantum Investment Fund?",
            "What is the current interest rate for savings accounts?"
        ]
    }
    
    # Knowledge base documents for RAG
    documents = [
        {
            "id": "doc1",
            "content": "The Quantum Investment Fund is a diversified equity fund with a focus on technology stocks. Annual management fee is 0.75%. Front-load fee is 2%. Minimum investment is $10,000.",
            "metadata": {"source": "Fund Prospectus", "date": "2024-01-01"}
        },
        {
            "id": "doc2", 
            "content": "Savings account interest rates vary by institution. Current national average is 0.45% APY. High-yield savings accounts may offer rates up to 4.5% APY.",
            "metadata": {"source": "Rate Survey", "date": "2024-12-01"}
        }
    ]
    
    # Initialize models
    logger.info("Initializing models...")
    models = {}
    
    try:
        models["gpt4_baseline"] = GPT4Baseline(config["models"]["baselines"]["gpt4"])
        logger.info("✓ GPT-4 baseline initialized")
    except Exception as e:
        logger.error(f"Failed to initialize GPT-4 baseline: {e}")
        return
    
    try:
        models["claude_baseline"] = ClaudeBaseline(config["models"]["baselines"]["claude"])
        logger.info("✓ Claude baseline initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize Claude baseline: {e}")
    
    try:
        framework_config = {
            **config["models"]["framework"],
            **config["framework"]
        }
        models["multilayer_framework"] = MultiLayeredAgent(framework_config)
        models["multilayer_framework"].initialize_knowledge_base(documents)
        logger.info("✓ Multi-layered framework initialized")
    except Exception as e:
        logger.error(f"Failed to initialize framework: {e}")
        return
    
    # Initialize SelfCheckGPT
    selfcheck = SelfCheckGPT()
    
    # Results storage
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_mode": True,
        "evaluations": {}
    }
    
    # Run limited evaluation
    for domain, queries in test_queries.items():
        logger.info(f"\nEvaluating domain: {domain}")
        domain_results = {}
        
        for model_name, model in models.items():
            logger.info(f"  Testing {model_name}...")
            
            # Test single query
            query = queries[0]
            try:
                response_data = model.generate_response(query)
                
                if response_data.get("response"):
                    logger.info(f"    ✓ Generated response for: '{query[:50]}...'")
                    domain_results[model_name] = {
                        "status": "success",
                        "sample_query": query,
                        "sample_response": response_data["response"][:200] + "...",
                        "confidence": response_data.get("confidence", 1.0),
                        "escalated": response_data.get("escalated", False)
                    }
                else:
                    logger.error(f"    ✗ No response generated")
                    domain_results[model_name] = {"status": "no_response"}
                    
            except Exception as e:
                logger.error(f"    ✗ Error: {e}")
                domain_results[model_name] = {"status": "error", "error": str(e)}
        
        results["evaluations"][domain] = domain_results
    
    # Save test results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"test_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nTest results saved to {output_file}")
    
    # Print summary
    print("\n=== Test Evaluation Summary ===")
    for domain, domain_results in results["evaluations"].items():
        print(f"\n{domain}:")
        for model, result in domain_results.items():
            status = result.get("status", "unknown")
            print(f"  {model}: {status}")
            if status == "success" and "multilayer_framework" in model:
                print(f"    - Confidence: {result.get('confidence', 'N/A')}")
                print(f"    - Escalated: {result.get('escalated', 'N/A')}")

if __name__ == "__main__":
    test_evaluation()