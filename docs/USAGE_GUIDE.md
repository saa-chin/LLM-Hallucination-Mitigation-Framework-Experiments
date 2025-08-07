# Usage Guide for Comparative Evaluation Framework

This guide provides step-by-step instructions for running the comparative evaluation of LLM hallucination mitigation approaches.

## Prerequisites

1. **Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set API Keys**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

## Running the Evaluation

### 1. Basic Evaluation

Run the complete comparative evaluation:

```bash
python scripts/run_evaluation.py --config config/eval_config.yaml
```

This will:
- Compare GPT-4 baseline, Claude baseline, and the multi-layered framework
- Use SelfCheckGPT to detect hallucinations
- Run TruthfulQA benchmark tests
- Save results to `results/evaluation_results_TIMESTAMP.json`

### 2. Analyze Results

After running the evaluation, analyze the results:

```bash
python scripts/analyze_results.py --results-dir results/
```

This generates:
- `results/analysis/summary.json` - Numerical summary
- `results/analysis/hallucination_comparison.png` - Bar chart comparison
- `results/analysis/confidence_distribution.png` - Confidence score analysis
- `results/analysis/evaluation_report.md` - Detailed markdown report

### 3. Quick Visualization

To only generate plots without full analysis:

```bash
python scripts/analyze_results.py --results-dir results/ --plot-only
```

## Customizing the Evaluation

### Modify Test Queries

Edit the test queries in `scripts/run_evaluation.py`:

```python
test_queries = {
    "financial_services": [
        "Your custom query here",
        # Add more queries
    ]
}
```

### Adjust Configuration

Edit `config/eval_config.yaml` to:
- Change model parameters (temperature, max_tokens)
- Adjust confidence thresholds
- Enable/disable benchmarks
- Modify RAG settings

### Add Knowledge Base Documents

For the multi-layered framework, add documents to the knowledge base:

```python
documents = [
    {
        "id": "unique_id",
        "content": "Your document content here",
        "metadata": {"source": "Document Source", "date": "2024-01-01"}
    }
]
```

## Understanding the Results

### Metrics Explained

1. **Hallucination Rate**: Percentage of responses flagged as containing hallucinations
2. **Mean Hallucination Score**: Average confidence in hallucination detection (0-1)
3. **TruthfulQA Accuracy**: Percentage of correct answers on TruthfulQA benchmark
4. **Confidence Scores**: Framework's self-reported confidence (baselines always 1.0)
5. **Escalation Rate**: Percentage of queries escalated to human review

### Expected Outcomes

- **Baselines**: Higher hallucination rates, no confidence estimation
- **Multi-layered Framework**: 
  - Lower hallucination rates due to RAG grounding
  - Confidence-based escalation for uncertain queries
  - Better performance on domain-specific queries

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure environment variables are set correctly
   - Check API key validity and quota

2. **Import Errors**
   - Make sure you're in the evaluation directory
   - Verify all dependencies are installed

3. **No Results Generated**
   - Check logs for errors
   - Ensure test queries are properly formatted
   - Verify knowledge base is initialized

### Debug Mode

For detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Batch Processing

For large-scale evaluation:

```python
# In run_evaluation.py
evaluator = ComparativeEvaluator(args.config)
evaluator.config["evaluation"]["batch_size"] = 50  # Increase batch size
evaluator.run_evaluation()
```

### Custom Benchmarks

To add custom benchmarks:

1. Create benchmark loader in `benchmark_evaluator.py`
2. Add configuration in `eval_config.yaml`
3. Implement evaluation logic

### Export Results

Convert results to different formats:

```python
# Export to CSV
df = pd.DataFrame(summary["metrics"])
df.to_csv("results.csv")

# Export to LaTeX
df.to_latex("results.tex")
```

## Next Steps

1. **Extend Test Coverage**: Add more diverse test queries
2. **Fine-tune Parameters**: Optimize confidence thresholds
3. **Add Benchmarks**: Integrate HALoGEN or other benchmarks
4. **Production Testing**: Test with real-world queries
5. **Performance Optimization**: Implement caching and parallel processing