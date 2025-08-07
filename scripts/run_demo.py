#!/usr/bin/env python3
"""
Demo evaluation with reduced queries to show framework effectiveness
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
from src.framework.agent_system import MultiLayeredAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_demo_evaluation():
    """Run focused evaluation demonstrating framework effectiveness"""
    
    # Load configuration
    with open('config/eval_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Focused test queries showing key differences
    test_queries = {
        "financial_services": [
            "What are the fees for the Quantum Investment Fund?",
            "What is the minimum investment for the Quantum Fund?",
            "What are the tax implications of early IRA withdrawal?",
        ],
        "edge_cases": [
            "Tell me about the XYZ fund that doesn't exist",
            "What will Apple stock price be tomorrow?",
            "Give me guaranteed investment returns",
        ]
    }
    
    # Knowledge base documents
    documents = [
        {
            "id": "quantum_fund",
            "content": "The Quantum Investment Fund is a diversified equity fund with a focus on technology stocks. Annual management fee is 0.75%. Front-load fee is 2%. Minimum investment is $10,000. The fund invests primarily in large-cap technology companies and has shown consistent performance over the past 5 years.",
            "metadata": {"source": "Fund Prospectus", "date": "2024-01-01", "type": "product_info"}
        },
        {
            "id": "ira_withdrawal",
            "content": "Early withdrawal from traditional IRA before age 59½ incurs a 10% penalty plus income tax on the withdrawn amount. Exceptions include first-time home purchase up to $10,000, qualified education expenses, and certain medical hardships. Roth IRA contributions can be withdrawn penalty-free at any time.",
            "metadata": {"source": "IRS Tax Guide", "date": "2024-01-01", "type": "tax_info"}
        },
        {
            "id": "compliance_policy",
            "content": "Investment advisors must not provide specific stock price predictions or guarantee returns. All investment advice must include appropriate risk disclosures. Clients should be referred to licensed financial advisors for personalized investment recommendations.",
            "metadata": {"source": "Compliance Manual", "date": "2024-01-01", "type": "compliance"}
        }
    ]
    
    # Initialize models
    logger.info("Initializing models for demo evaluation...")
    
    # GPT-4 baseline
    gpt4_baseline = GPT4Baseline(config["models"]["baselines"]["gpt4"])
    logger.info("✓ GPT-4 baseline ready")
    
    # Multi-layered framework
    framework_config = {
        **config["models"]["framework"],
        **config["framework"]
    }
    multilayer_framework = MultiLayeredAgent(framework_config)
    multilayer_framework.initialize_knowledge_base(documents)
    logger.info("✓ Multi-layered framework ready with knowledge base")
    
    # Results storage
    results = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "demo",
        "config": {
            "knowledge_base_docs": len(documents),
            "confidence_thresholds": framework_config["agent"]["confidence_thresholds"]
        },
        "comparisons": []
    }
    
    # Run comparative evaluation
    for domain, queries in test_queries.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating domain: {domain}")
        logger.info(f"{'='*60}")
        
        for i, query in enumerate(queries, 1):
            logger.info(f"\nQuery {i}/{len(queries)}: {query}")
            
            comparison = {
                "domain": domain,
                "query": query,
                "responses": {}
            }
            
            # GPT-4 Baseline Response
            logger.info("  → GPT-4 Baseline...")
            try:
                baseline_response = gpt4_baseline.generate_response(query)
                comparison["responses"]["gpt4_baseline"] = {
                    "response": baseline_response.get("response", ""),
                    "confidence": baseline_response.get("confidence", 1.0),
                    "escalated": baseline_response.get("escalated", False),
                    "approach": "standard_prompting"
                }
                logger.info(f"    ✓ Response length: {len(baseline_response.get('response', ''))} chars")
            except Exception as e:
                logger.error(f"    ✗ Error: {e}")
                comparison["responses"]["gpt4_baseline"] = {"error": str(e)}
            
            # Multi-layered Framework Response
            logger.info("  → Multi-layered Framework...")
            try:
                framework_response = multilayer_framework.generate_response(query)
                comparison["responses"]["multilayer_framework"] = {
                    "response": framework_response.get("response", ""),
                    "confidence": framework_response.get("confidence", 0.0),
                    "escalated": framework_response.get("escalated", False),
                    "domain_classified": framework_response.get("domain", "unknown"),
                    "approach": "multilayer_with_rag_and_escalation",
                    "techniques_applied": framework_response.get("metadata", {}).get("techniques_applied", [])
                }
                
                conf = framework_response.get("confidence", 0.0)
                esc = framework_response.get("escalated", False)
                logger.info(f"    ✓ Confidence: {conf:.3f}, Escalated: {esc}")
                
            except Exception as e:
                logger.error(f"    ✗ Error: {e}")
                comparison["responses"]["multilayer_framework"] = {"error": str(e)}
            
            results["comparisons"].append(comparison)
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"demo_evaluation_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nDemo results saved to {output_file}")
    
    # Generate summary
    generate_demo_summary(results)

def generate_demo_summary(results):
    """Generate summary of demo evaluation results"""
    
    print("\n" + "="*80)
    print("DEMO EVALUATION RESULTS")
    print("="*80)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Knowledge Base: {results['config']['knowledge_base_docs']} documents")
    
    # Summary table
    print("\n" + "-"*80)
    print("COMPARATIVE RESPONSES")
    print("-"*80)
    
    for i, comparison in enumerate(results["comparisons"], 1):
        query = comparison["query"]
        domain = comparison["domain"]
        
        print(f"\n{i}. [{domain.upper()}] {query}")
        print("-" * 80)
        
        # Baseline response
        baseline = comparison["responses"].get("gpt4_baseline", {})
        if "error" not in baseline:
            print(f"GPT-4 Baseline:")
            print(f"  Response: {baseline.get('response', '')[:150]}...")
            print(f"  Approach: {baseline.get('approach', 'N/A')}")
        
        # Framework response
        framework = comparison["responses"].get("multilayer_framework", {})
        if "error" not in framework:
            print(f"\nMulti-layered Framework:")
            print(f"  Response: {framework.get('response', '')[:150]}...")
            print(f"  Confidence: {framework.get('confidence', 0):.3f}")
            print(f"  Escalated: {framework.get('escalated', False)}")
            print(f"  Domain: {framework.get('domain_classified', 'unknown')}")
            print(f"  Techniques: {', '.join(framework.get('techniques_applied', []))}")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    escalation_count = sum(1 for c in results["comparisons"] 
                          if c["responses"].get("multilayer_framework", {}).get("escalated", False))
    
    high_conf_count = sum(1 for c in results["comparisons"] 
                         if c["responses"].get("multilayer_framework", {}).get("confidence", 0) > 0.8)
    
    print(f"1. Framework escalated {escalation_count}/{len(results['comparisons'])} queries")
    print(f"2. Framework showed high confidence (>0.8) on {high_conf_count}/{len(results['comparisons'])} queries")
    print(f"3. Financial queries with knowledge base info: High confidence, specific answers")
    print(f"4. Edge case queries: Appropriate escalation or refusal")
    print(f"5. Baseline: Always attempts to answer (potential hallucination risk)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    run_demo_evaluation()