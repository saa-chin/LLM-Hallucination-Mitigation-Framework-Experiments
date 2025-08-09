# Multi-Layered Framework for LLM Hallucination Mitigation in High-Stakes Applications

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/multi-layered-llm-hallucination-mitigation/blob/main/LLM_Hallucination_Mitigation_Demo.ipynb)
[![Paper](https://img.shields.io/badge/Paper-Computers_2025-red)](https://doi.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)

## Try It Now - No Installation Required!

Click the "Open in Colab" button above to run this framework directly in your browser. Just add your OpenAI API key and start experimenting!

## Quick Start in 30 Seconds

```python
# 1. Install (run in Colab or locally)
!pip install openai chromadb numpy tiktoken

# 2. Set your API key
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# 3. Run the framework
from framework import HallucinationMitigationFramework
framework = HallucinationMitigationFramework()
result = framework.process_query("What are the fees for the Quantum Fund?")
print(result['response'])
```

## About This Project

Large language models have transformed how we interact with AI, but their tendency to generate confident yet incorrect information—commonly known as hallucination—remains a critical challenge. This is especially problematic in high-stakes domains like financial services, where a single incorrect statement about fees or regulations could lead to regulatory violations or client lawsuits.

This repository implements a comprehensive framework that layers multiple mitigation strategies to address this challenge. Rather than relying on any single technique, we combine structured prompt design, retrieval-augmented generation with verifiable evidence sources, and targeted confidence-based escalation mechanisms.

## Interactive Demo

We provide a fully interactive Jupyter notebook that you can run directly in Google Colab:

1. **[Open the Interactive Demo](https://colab.research.google.com/github/yourusername/multi-layered-llm-hallucination-mitigation/blob/main/LLM_Hallucination_Mitigation_Demo.ipynb)**
2. Add your OpenAI API key when prompted
3. Run the cells to see the framework in action
4. Try your own queries and experiment with the settings

### What's in the Notebook?

- **Live API Integration**: Connect directly to OpenAI GPT-4
- **Interactive Examples**: Test with pre-loaded financial queries
- **Real-time Metrics**: See confidence scores and domain classification
- **Comparison Mode**: Compare framework responses with baseline GPT-4
- **Custom Knowledge Base**: Add your own documents and test immediately
- **Performance Visualization**: Generate charts showing framework effectiveness

## Key Results

Our evaluation demonstrates significant improvements over baseline GPT-4:

![Comprehensive Evaluation Results](results/figure1_comprehensive_evaluation.png)
*Figure 1: Framework performance across multiple dimensions showing 51% overall accuracy improvement*

- **89% overall accuracy** compared to 59% baseline (p < 0.001, Cohen's d = 1.73)
- **100% accuracy in financial services domain** versus 33% baseline
- **82% reduction in response length** while maintaining higher accuracy
- **Zero hallucinations detected** in evaluation dataset
- **12% appropriate escalation rate** for uncertain queries

![Case Studies](results/figure2_case_studies.png)
*Figure 2: Real-world case studies demonstrating hallucination prevention and appropriate escalation*

## Framework Architecture

Our approach integrates three complementary layers:

### 1. Foundational Layer: Prompt Engineering
```python
# Example: Few-shot prompting
prompt = PromptEngineer.create_few_shot_prompt(
    query="What are the fund fees?",
    examples=[
        {"question": "What is the rate?", "answer": "0.75% annually"},
        {"question": "Any other fees?", "answer": "2% front-load"}
    ]
)
```

### 2. Architectural Layer: RAG
```python
# Example: Retrieve relevant documents
docs, confidence = rag_system.retrieve(
    query="What are the fees?",
    top_k=5,
    threshold=0.75
)
```

### 3. Behavioral Layer: Confidence-Based Escalation
```python
# Example: Smart escalation
if confidence < domain_threshold:
    return escalate_to_human()
else:
    return generate_grounded_response()
```

## Installation for Local Development

If you prefer to run locally instead of using Colab:

```bash
# Clone the repository
git clone https://github.com/yourusername/multi-layered-llm-hallucination-mitigation.git
cd multi-layered-llm-hallucination-mitigation

# Install dependencies
pip install -r requirements.txt

# Set up your API key
export OPENAI_API_KEY="your-openai-api-key"

# Run the demo
python scripts/run_demo.py
```

## Performance Comparison

### Domain-Specific Accuracy

| Domain | GPT-4 Baseline | Our Framework | Improvement | Statistical Significance |
|--------|----------------|---------------|-------------|--------------------------|
| Financial Services | 33% | 100% | +200% | p < 0.01 |
| Edge Cases | 67% | 83% | +24% | p < 0.05 |
| General Knowledge | 78% | 85% | +9% | p = 0.15 |
| **Overall** | **59%** | **89%** | **+51%** | **p < 0.001** |

### Response Characteristics

| Metric | GPT-4 Baseline | Our Framework | Impact |
|--------|----------------|---------------|--------|
| Mean Response Length | 1,611 characters | 289 characters | 82% more concise |
| Appropriate Escalations | 0% | 12% | Risk management enabled |
| Confidence Calibration | Fixed at 1.0 | Dynamic 0.76-0.88 | Uncertainty quantification |
| Hallucination Incidents | 3 detected | 0 detected | Complete prevention |

## Configuration

The framework is highly configurable. Edit settings in the notebook or use `config/eval_config.yaml`:

```python
# In the notebook, you can modify these directly:
DOMAIN_CONFIGS = {
    Domain.FINANCIAL: DomainConfig(
        confidence_threshold=0.80,
        keywords=["fee", "investment", "fund"]
    ),
    Domain.COMPLIANCE: DomainConfig(
        confidence_threshold=0.85,
        keywords=["regulation", "compliance"]
    )
}
```

## Paper Reference

**Multi-Layered Framework for LLM Hallucination Mitigation in High-Stakes Applications: A Tutorial**  
*Sachin Hiriyanna (Navan Inc.) and Wenbing Zhao (Cleveland State University)*  
*Computers, MDPI, 2025*

If you use this framework in your research or production systems, please cite:
```bibtex
@article{hiriyanna2025multilayered,
    title={Multi-Layered Framework for LLM Hallucination Mitigation in High-Stakes Applications: A Tutorial},
    author={Hiriyanna, Sachin and Zhao, Wenbing},
    journal={Computers},
    publisher={MDPI},
    year={2025},
    doi={10.3390/computers}
}
```

## Project Structure

```
├── LLM_Hallucination_Mitigation_Demo.ipynb  # Interactive notebook (start here!)
├── src/
│   ├── framework/
│   │   ├── agent_system.py                  # Main orchestrator
│   │   ├── rag_layer.py                     # RAG implementation
│   │   └── prompt_layer.py                  # Prompt engineering
│   └── evaluation/
│       └── benchmark_evaluator.py           # Evaluation framework
├── data/
│   ├── financial_queries.json               # Test queries
│   └── knowledge_base.json                  # Sample documents
├── results/
│   ├── figure1_comprehensive_evaluation.png # Performance charts
│   └── figure2_case_studies.png            # Case study results
└── config/
    └── eval_config.yaml                     # Configuration settings
```

## Implementation Guidelines

### For Production Deployment

1. **Start with the notebook** - Experiment with your use case
2. **Customize the knowledge base** - Add your domain-specific documents
3. **Tune confidence thresholds** - Adjust based on your risk tolerance
4. **Monitor performance** - Track metrics over time

### Security Considerations

The framework includes several security features:
- Input validation to prevent prompt injection
- Document integrity verification
- Rate limiting capabilities
- Audit trail generation

## Extending the Framework

### Add Your Own Knowledge Base

In the notebook, you can easily add custom documents:

```python
# Add directly in the notebook
new_doc = {
    "content": "Your company policy text here...",
    "metadata": {"source": "Policy Manual", "type": "compliance"}
}
rag_system.documents.append(new_doc)
```

### Customize Domain Classification

```python
# Modify domain keywords in the notebook
DOMAIN_CONFIGS[Domain.FINANCIAL].keywords.extend([
    "portfolio", "asset", "equity"
])
```

## Running Tests

Execute the test suite:
```bash
python tests/test_framework.py
```

Or run tests in the notebook:
```python
# Cell in notebook
!python tests/test_framework.py
```

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

We thank the teams at OpenAI and Anthropic for API access that made this research possible. The work builds on foundational research in prompt engineering, retrieval-augmented generation, and AI safety from the broader research community.

## Contact

For questions about implementation or the research paper:
- Sachin Hiriyanna: sachinh@ieee.org
- Wenbing Zhao: wenbing@ieee.org

For technical issues, please open a GitHub issue.

---

## Quick Links

- **[Interactive Demo (Colab)](https://colab.research.google.com/github/yourusername/multi-layered-llm-hallucination-mitigation/blob/main/LLM_Hallucination_Mitigation_Demo.ipynb)** - Start here!
- **[Paper](https://doi.org/)** - Full research paper
- **[Issues](https://github.com/yourusername/multi-layered-llm-hallucination-mitigation/issues)** - Report bugs or request features
- **[Discussions](https://github.com/yourusername/multi-layered-llm-hallucination-mitigation/discussions)** - Community discussions