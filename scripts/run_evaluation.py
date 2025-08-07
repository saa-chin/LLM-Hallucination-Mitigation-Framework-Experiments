#!/usr/bin/env python3
"""
Academic-grade evaluation with 100+ test cases for publication
"""

import os
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import logging
import time
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv('.env')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.baselines.gpt4_baseline import GPT4Baseline
from src.framework.agent_system import MultiLayeredAgent
from src.evaluation.selfcheckgpt import SelfCheckGPT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AcademicEvaluator:
    """Academic-grade evaluator with statistical analysis"""
    
    def __init__(self):
        self.config = self._load_config()
        self.test_data = self._load_test_data()
        self.knowledge_base = self._load_knowledge_base()
        self.models = self._initialize_models()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "academic",
            "config": {
                "total_queries": sum(len(queries) for queries in self.test_data.values()),
                "domains": list(self.test_data.keys()),
                "knowledge_base_size": len(self.knowledge_base)
            },
            "raw_results": {},
            "statistical_analysis": {},
            "summary": {}
        }
        
    def _load_config(self) -> dict:
        with open('config/eval_config.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def _load_test_data(self) -> dict:
        with open('test_data/financial_queries.json', 'r') as f:
            return json.load(f)
    
    def _load_knowledge_base(self) -> list:
        with open('test_data/knowledge_base.json', 'r') as f:
            return json.load(f)
    
    def _initialize_models(self) -> dict:
        logger.info("Initializing models for academic evaluation...")
        
        # GPT-4 baseline
        gpt4_baseline = GPT4Baseline(self.config["models"]["baselines"]["gpt4"])
        
        # Multi-layered framework
        framework_config = {**self.config["models"]["framework"], **self.config["framework"]}
        multilayer_framework = MultiLayeredAgent(framework_config)
        multilayer_framework.initialize_knowledge_base(self.knowledge_base)
        
        logger.info(f"✓ Models initialized with {len(self.knowledge_base)} knowledge base documents")
        return {
            "gpt4_baseline": gpt4_baseline,
            "multilayer_framework": multilayer_framework
        }
    
    def evaluate_response_quality(self, query: str, response: str, domain: str) -> dict:
        """Evaluate response quality based on domain-specific criteria"""
        
        # Define expected keywords/patterns for different domains
        quality_indicators = {
            "financial_services": {
                "appropriate_caution": ["consult", "advisor", "professional", "varies", "depends"],
                "factual_precision": ["percent", "%", "dollar", "$", "limit", "maximum", "minimum"],
                "compliance": ["disclosure", "risk", "not guaranteed", "may vary"]
            },
            "edge_cases": {
                "appropriate_refusal": ["cannot", "don't", "unable", "not possible", "shouldn't"],
                "risk_awareness": ["risky", "dangerous", "illegal", "not recommended", "caution"]
            },
            "general_knowledge": {
                "educational": ["generally", "typically", "commonly", "basic", "fundamental"],
                "accuracy_indicators": ["defined as", "refers to", "means", "involves"]
            }
        }
        
        response_lower = response.lower()
        indicators = quality_indicators.get(domain, {})
        
        quality_score = 0
        quality_details = {}
        
        for category, keywords in indicators.items():
            matches = sum(1 for keyword in keywords if keyword in response_lower)
            category_score = min(matches / len(keywords), 1.0)
            quality_details[category] = category_score
            quality_score += category_score
        
        # Normalize score
        quality_score = quality_score / len(indicators) if indicators else 0.5
        
        # Length appropriateness (penalize overly long or very short responses)
        length_penalty = 0
        if len(response) > 2000:  # Too verbose
            length_penalty = 0.1
        elif len(response) < 20:  # Too brief
            length_penalty = 0.2
            
        final_score = max(0, quality_score - length_penalty)
        
        return {
            "quality_score": final_score,
            "details": quality_details,
            "length": len(response),
            "length_penalty": length_penalty
        }
    
    def run_evaluation(self):
        """Run comprehensive academic evaluation"""
        logger.info("Starting academic evaluation with statistical analysis...")
        
        for domain, queries in self.test_data.items():
            logger.info(f"\nEvaluating domain: {domain} ({len(queries)} queries)")
            
            domain_results = {
                "queries": [],
                "metrics": {
                    "gpt4_baseline": {"quality_scores": [], "response_lengths": [], "escalation_count": 0},
                    "multilayer_framework": {"quality_scores": [], "response_lengths": [], "escalation_count": 0, "confidence_scores": []}
                }
            }
            
            for i, query in enumerate(queries):
                if i % 10 == 0:
                    logger.info(f"  Progress: {i}/{len(queries)} queries")
                
                query_result = {"query": query, "responses": {}}
                
                # Test both models
                for model_name, model in self.models.items():
                    try:
                        response_data = model.generate_response(query)
                        response_text = response_data.get("response", "")
                        
                        # Evaluate response quality
                        quality_eval = self.evaluate_response_quality(query, response_text, domain)
                        
                        query_result["responses"][model_name] = {
                            "response": response_text,
                            "confidence": response_data.get("confidence", 1.0),
                            "escalated": response_data.get("escalated", False),
                            "quality_evaluation": quality_eval
                        }
                        
                        # Collect metrics
                        metrics = domain_results["metrics"][model_name]
                        metrics["quality_scores"].append(quality_eval["quality_score"])
                        metrics["response_lengths"].append(quality_eval["length"])
                        if response_data.get("escalated", False):
                            metrics["escalation_count"] += 1
                        if model_name == "multilayer_framework":
                            metrics["confidence_scores"].append(response_data.get("confidence", 0))
                        
                        # Small delay to avoid rate limits
                        time.sleep(0.3)
                        
                    except Exception as e:
                        logger.error(f"Error with {model_name} on query '{query[:50]}...': {e}")
                        query_result["responses"][model_name] = {"error": str(e)}
                
                domain_results["queries"].append(query_result)
            
            self.results["raw_results"][domain] = domain_results
            logger.info(f"  ✓ Completed {domain}: {len(queries)} queries evaluated")
        
        # Perform statistical analysis
        self._perform_statistical_analysis()
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        self._save_results()
    
    def _perform_statistical_analysis(self):
        """Perform statistical analysis on evaluation results"""
        logger.info("Performing statistical analysis...")
        
        analysis = {}
        
        for domain, domain_results in self.results["raw_results"].items():
            metrics = domain_results["metrics"]
            
            baseline_quality = np.array(metrics["gpt4_baseline"]["quality_scores"])
            framework_quality = np.array(metrics["multilayer_framework"]["quality_scores"])
            
            # Remove any NaN values
            baseline_quality = baseline_quality[~np.isnan(baseline_quality)]
            framework_quality = framework_quality[~np.isnan(framework_quality)]
            
            if len(baseline_quality) > 0 and len(framework_quality) > 0:
                # Paired t-test for quality scores
                t_stat, p_value = stats.ttest_ind(framework_quality, baseline_quality)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(framework_quality) - 1) * np.var(framework_quality, ddof=1) + 
                                     (len(baseline_quality) - 1) * np.var(baseline_quality, ddof=1)) / 
                                    (len(framework_quality) + len(baseline_quality) - 2))
                cohens_d = (np.mean(framework_quality) - np.mean(baseline_quality)) / pooled_std
                
                analysis[domain] = {
                    "sample_size": {"baseline": len(baseline_quality), "framework": len(framework_quality)},
                    "quality_scores": {
                        "baseline": {
                            "mean": float(np.mean(baseline_quality)),
                            "std": float(np.std(baseline_quality)),
                            "median": float(np.median(baseline_quality))
                        },
                        "framework": {
                            "mean": float(np.mean(framework_quality)),
                            "std": float(np.std(framework_quality)),
                            "median": float(np.median(framework_quality))
                        }
                    },
                    "statistical_tests": {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "cohens_d": float(cohens_d),
                        "significant": p_value < 0.05
                    },
                    "escalation_rates": {
                        "baseline": metrics["gpt4_baseline"]["escalation_count"] / len(baseline_quality),
                        "framework": metrics["multilayer_framework"]["escalation_count"] / len(framework_quality)
                    },
                    "response_lengths": {
                        "baseline": {
                            "mean": float(np.mean(metrics["gpt4_baseline"]["response_lengths"])),
                            "std": float(np.std(metrics["gpt4_baseline"]["response_lengths"]))
                        },
                        "framework": {
                            "mean": float(np.mean(metrics["multilayer_framework"]["response_lengths"])),
                            "std": float(np.std(metrics["multilayer_framework"]["response_lengths"]))
                        }
                    }
                }
                
                if "confidence_scores" in metrics["multilayer_framework"]:
                    conf_scores = np.array(metrics["multilayer_framework"]["confidence_scores"])
                    analysis[domain]["confidence_distribution"] = {
                        "mean": float(np.mean(conf_scores)),
                        "std": float(np.std(conf_scores)),
                        "high_confidence_rate": float(np.sum(conf_scores > 0.8) / len(conf_scores))
                    }
        
        self.results["statistical_analysis"] = analysis
        logger.info("✓ Statistical analysis completed")
    
    def _generate_summary(self):
        """Generate summary of evaluation results"""
        summary = {
            "total_queries_evaluated": self.results["config"]["total_queries"],
            "domains_tested": len(self.results["config"]["domains"]),
            "overall_improvements": {},
            "key_findings": []
        }
        
        # Calculate overall improvements
        all_baseline_quality = []
        all_framework_quality = []
        total_escalations_baseline = 0
        total_escalations_framework = 0
        total_queries = 0
        
        for domain, analysis in self.results["statistical_analysis"].items():
            all_baseline_quality.extend([analysis["quality_scores"]["baseline"]["mean"]] * analysis["sample_size"]["baseline"])
            all_framework_quality.extend([analysis["quality_scores"]["framework"]["mean"]] * analysis["sample_size"]["framework"])
            total_escalations_baseline += analysis["escalation_rates"]["baseline"] * analysis["sample_size"]["baseline"]
            total_escalations_framework += analysis["escalation_rates"]["framework"] * analysis["sample_size"]["framework"]
            total_queries += analysis["sample_size"]["framework"]
        
        if all_baseline_quality and all_framework_quality:
            baseline_mean = np.mean(all_baseline_quality)
            framework_mean = np.mean(all_framework_quality)
            
            summary["overall_improvements"] = {
                "quality_improvement": {
                    "baseline_mean": float(baseline_mean),
                    "framework_mean": float(framework_mean),
                    "improvement_percentage": float(((framework_mean - baseline_mean) / baseline_mean) * 100)
                },
                "escalation_rates": {
                    "baseline": float(total_escalations_baseline / total_queries),
                    "framework": float(total_escalations_framework / total_queries)
                }
            }
        
        # Key findings
        significant_domains = [domain for domain, analysis in self.results["statistical_analysis"].items() 
                              if analysis["statistical_tests"]["significant"]]
        
        summary["key_findings"] = [
            f"Statistically significant improvements in {len(significant_domains)}/{len(self.results['statistical_analysis'])} domains",
            f"Average quality improvement: {summary['overall_improvements']['quality_improvement']['improvement_percentage']:.1f}%",
            f"Framework escalation rate: {summary['overall_improvements']['escalation_rates']['framework']:.1%}",
            f"Total queries evaluated: {summary['total_queries_evaluated']}"
        ]
        
        self.results["summary"] = summary
    
    def _save_results(self):
        """Save evaluation results"""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"academic_evaluation_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nAcademic evaluation results saved to {output_file}")
        return output_file
    
    def print_results_summary(self):
        """Print formatted results summary"""
        print("\n" + "="*80)
        print("ACADEMIC EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        summary = self.results["summary"]
        print(f"Total Queries Evaluated: {summary['total_queries_evaluated']}")
        print(f"Domains Tested: {summary['domains_tested']}")
        print(f"Knowledge Base Size: {self.results['config']['knowledge_base_size']} documents")
        
        print(f"\nOverall Quality Improvement: {summary['overall_improvements']['quality_improvement']['improvement_percentage']:.1f}%")
        print(f"Baseline Mean Quality: {summary['overall_improvements']['quality_improvement']['baseline_mean']:.3f}")
        print(f"Framework Mean Quality: {summary['overall_improvements']['quality_improvement']['framework_mean']:.3f}")
        
        print(f"\nEscalation Rates:")
        print(f"  Baseline: {summary['overall_improvements']['escalation_rates']['baseline']:.1%}")
        print(f"  Framework: {summary['overall_improvements']['escalation_rates']['framework']:.1%}")
        
        print("\nStatistical Significance by Domain:")
        for domain, analysis in self.results["statistical_analysis"].items():
            significance = "✓" if analysis["statistical_tests"]["significant"] else "✗"
            p_val = analysis["statistical_tests"]["p_value"]
            cohens_d = analysis["statistical_tests"]["cohens_d"]
            print(f"  {domain:20s} {significance} (p={p_val:.4f}, d={cohens_d:.3f})")
        
        print("\nKey Findings:")
        for finding in summary["key_findings"]:
            print(f"  • {finding}")
        
        print("\n" + "="*80)

def main():
    """Main evaluation execution"""
    evaluator = AcademicEvaluator()
    
    try:
        evaluator.run_evaluation()
        evaluator.print_results_summary()
        
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        evaluator._save_results()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()