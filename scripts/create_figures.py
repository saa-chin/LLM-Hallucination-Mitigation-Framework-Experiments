#!/usr/bin/env python3
"""
Clean, Readable Figure Generation for Academic Paper
Large text, light backgrounds, minimal clutter
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from matplotlib import rcParams

# Configure matplotlib for maximum readability
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
rcParams['font.size'] = 14  # Larger base font
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['figure.titlesize'] = 18

# Clean color palette - high contrast, simple
COLORS = {
    'framework': '#2E86AB',     # Clear blue
    'baseline': '#A23B72',      # Clear purple-red
    'background': '#F8F9FA',    # Very light gray
    'text': '#000000',          # Pure black
    'success': '#28A745',       # Clear green
    'warning': '#FFC107',       # Clear yellow
    'light_bg': '#FFFFFF',      # Pure white
    'grid': '#DEE2E6',          # Light gray for grids
}

def create_simple_text_box(ax, x, y, width, height, text, bgcolor='white', 
                          textcolor='black', fontsize=12, fontweight='bold'):
    """Create clean text box with light background and dark text"""
    # Simple rectangle background
    bbox = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.01",
        facecolor=bgcolor,
        edgecolor=textcolor,
        linewidth=1.5,
        alpha=0.9
    )
    ax.add_patch(bbox)
    
    # Large, clear text
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center', fontsize=fontsize,
           color=textcolor, weight=fontweight,
           wrap=True)

def create_figure1_clean():
    """Create clean, readable Figure 1"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor(COLORS['background'])
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92, bottom=0.08, left=0.08, right=0.95)
    
    # Large, clear title
    fig.suptitle('Multi-Layered Framework: Comprehensive Evaluation Results',
                fontsize=20, fontweight='bold', color=COLORS['text'], y=0.96)
    
    # (a) Response Accuracy - Simplified
    ax1 = axes[0, 0]
    domains = ['Financial', 'Edge Cases', 'General', 'Overall']
    baseline_acc = [33, 67, 78, 59]
    framework_acc = [100, 83, 85, 89]
    
    x = np.arange(len(domains))
    width = 0.4
    
    bars1 = ax1.bar(x - width/2, baseline_acc, width, label='Baseline (GPT-4)', 
                   color=COLORS['baseline'], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, framework_acc, width, label='Framework',
                   color=COLORS['framework'], alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=14)
    ax1.set_title('(a) Accuracy by Domain', fontweight='bold', pad=20, fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains, fontsize=12)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('white')
    ax1.set_ylim(0, 110)
    
    # Large value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(height)}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    
    # (b) Key Metrics - Simplified
    ax2 = axes[0, 1]
    metrics = ['Accuracy\nImproved', 'Length\nReduced', 'Hallucinations\nPrevented']
    values = [51, 82, 100]
    colors = [COLORS['framework'], COLORS['success'], COLORS['success']]
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1)
    ax2.set_ylabel('Percentage (%)', fontweight='bold', fontsize=14)
    ax2.set_title('(b) Key Performance Gains', fontweight='bold', pad=20, fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('white')
    ax2.set_ylim(0, 110)
    
    # Large value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(height)}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    
    # (c) Statistical Significance - Clean
    ax3 = axes[0, 2]
    domains_stats = ['Financial', 'Overall']
    effect_sizes = [2.84, 1.73]
    p_values = ['p < 0.01', 'p < 0.001']
    
    bars = ax3.barh(domains_stats, effect_sizes, color=COLORS['success'], alpha=0.8, 
                   edgecolor='black', linewidth=1)
    ax3.set_xlabel("Effect Size (Cohen's d)", fontweight='bold', fontsize=14)
    ax3.set_title('(c) Statistical Impact', fontweight='bold', pad=20, fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('white')
    
    # Large effect size labels
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        width = bar.get_width()
        ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'd = {width:.2f}\n{p_val}', ha='left', va='center', 
                fontweight='bold', fontsize=12)
    
    # Add interpretation lines
    ax3.axvline(x=0.8, color='gray', linestyle='--', alpha=0.6)
    ax3.text(0.8, 1.5, 'Large Effect', rotation=90, ha='center', va='bottom', fontsize=10)
    ax3.set_xlim(0, 3.2)
    
    # (d) Response Length Comparison - Simple
    ax4 = axes[1, 0]
    methods = ['Baseline', 'Framework']
    lengths = [1611, 289]
    colors_bars = [COLORS['baseline'], COLORS['framework']]
    
    bars = ax4.bar(methods, lengths, color=colors_bars, alpha=0.8, 
                  edgecolor='black', linewidth=1)
    ax4.set_ylabel('Response Length (chars)', fontweight='bold', fontsize=14)
    ax4.set_title('(d) Response Conciseness', fontweight='bold', pad=20, fontsize=16)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('white')
    
    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    
    # Improvement annotation
    ax4.annotate('82% Shorter', xy=(1, 289), xytext=(0.5, 800),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=3),
                fontsize=14, fontweight='bold', color=COLORS['success'],
                ha='center')
    
    # (e) Confidence Distribution - Simplified
    ax5 = axes[1, 1]
    conf_labels = ['Low\n(Escalated)', 'Medium', 'High']
    conf_values = [12, 63, 25]
    conf_colors = [COLORS['warning'], COLORS['framework'], COLORS['success']]
    
    wedges, texts, autotexts = ax5.pie(conf_values, labels=conf_labels, autopct='%1.0f%%',
                                      colors=conf_colors, startangle=90, 
                                      textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # Make text more visible
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    ax5.set_title('(e) Confidence Levels', fontweight='bold', pad=20, fontsize=16)
    
    # (f) Simple Framework Flow - Much Cleaner
    ax6 = axes[1, 2]
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    ax6.set_title('(f) Framework Components', fontweight='bold', pad=20, fontsize=16)
    ax6.set_facecolor('white')
    
    # Simple vertical flow - much cleaner
    steps = [
        {'name': 'Query Input', 'pos': (5, 8.5), 'color': '#E9ECEF'},
        {'name': 'Domain Classification', 'pos': (5, 7), 'color': '#E9ECEF'}, 
        {'name': 'RAG + Prompts', 'pos': (5, 5.5), 'color': '#E9ECEF'},
        {'name': 'Confidence Check', 'pos': (5, 4), 'color': '#E9ECEF'},
        {'name': 'Response/Escalate', 'pos': (5, 2.5), 'color': '#E9ECEF'}
    ]
    
    for step in steps:
        create_simple_text_box(ax6, step['pos'][0]-1.5, step['pos'][1]-0.4, 3, 0.8,
                              step['name'], bgcolor=step['color'], textcolor='black', 
                              fontsize=12, fontweight='bold')
    
    # Simple arrows between steps
    arrow_pairs = [
        ((5, 8.1), (5, 7.4)),   # Input to Classification
        ((5, 6.6), (5, 5.9)),   # Classification to RAG
        ((5, 5.1), (5, 4.4)),   # RAG to Confidence
        ((5, 3.6), (5, 2.9)),   # Confidence to Response
    ]
    
    for start, end in arrow_pairs:
        ax6.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['text']))
    
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/sachin/Documents/GitHub/A-Practical-Guide-to-LLMs-Book/paper_mitigate_hallucination/initialsub/figure1_comprehensive_evaluation.png', 
                dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

def create_figure2_clean():
    """Create clean, readable Figure 2 with large text"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(COLORS['background'])
    plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92, bottom=0.08, left=0.08, right=0.95)
    
    # Large title
    fig.suptitle('Hallucination Prevention: Real Case Studies',
                fontsize=20, fontweight='bold', color=COLORS['text'], y=0.96)
    
    # (a) Financial Query Case - Much Cleaner
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title('(a) Financial Query Results', fontweight='bold', fontsize=16, pad=20)
    ax1.set_facecolor('white')
    
    # Query
    create_simple_text_box(ax1, 1, 8.5, 8, 1, 
                          'Query: "Quantum Investment Fund fees?"',
                          bgcolor='#F8F9FA', textcolor='black', fontsize=12)
    
    # Baseline result - simple
    create_simple_text_box(ax1, 0.5, 6.5, 4, 1.5,
                          'GPT-4 Baseline:\n"I don\'t have specific info..."\n1,917 characters',
                          bgcolor='#FFE6E6', textcolor='black', fontsize=11)
    
    # Framework result - simple
    create_simple_text_box(ax1, 5.5, 6.5, 4, 1.5,
                          'Framework:\n"Annual fee 0.75%, load fee 2%"\n91 characters (95% shorter)',
                          bgcolor='#E6F3FF', textcolor='black', fontsize=11)
    
    # Result
    create_simple_text_box(ax1, 2, 4, 6, 1.5,
                          'RESULT: Precise, accurate answer\nwith 95% length reduction',
                          bgcolor='#E6FFE6', textcolor='black', fontsize=12)
    
    # Simple arrow
    ax1.annotate('BETTER', xy=(7.5, 6.0), xytext=(2.5, 6.0),
                arrowprops=dict(arrowstyle='->', lw=4, color=COLORS['success']),
                fontsize=14, fontweight='bold', color=COLORS['success'], ha='center')
    
    ax1.axis('off')
    
    # (b) Hallucination Prevention - Cleaner
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title('(b) Prevented Hallucination', fontweight='bold', fontsize=16, pad=20)
    ax2.set_facecolor('white')
    
    # Query
    create_simple_text_box(ax2, 1, 8.5, 8, 1,
                          'Query: "Quantum Fund minimum investment?"',
                          bgcolor='#F8F9FA', textcolor='black', fontsize=12)
    
    # Baseline hallucination
    create_simple_text_box(ax2, 0.5, 6, 4, 2,
                          'GPT-4 HALLUCINATION:\nConfused with Soros fund\n"Not open to public"\nWRONG INFORMATION',
                          bgcolor='#FFE6E6', textcolor='black', fontsize=11)
    
    # Framework correct
    create_simple_text_box(ax2, 5.5, 6, 4, 2,
                          'Framework CORRECT:\n"Minimum investment $10,000"\nFrom knowledge base\nACCURATE',
                          bgcolor='#E6F3FF', textcolor='black', fontsize=11)
    
    # Prevention highlight
    create_simple_text_box(ax2, 1.5, 3, 7, 1.5,
                          'HALLUCINATION PREVENTED\nFramework used correct source',
                          bgcolor='#E6FFE6', textcolor='black', fontsize=12)
    
    ax2.axis('off')
    
    # (c) Risk Management - Cleaner
    ax3 = axes[1, 0]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.set_title('(c) Appropriate Escalation', fontweight='bold', fontsize=16, pad=20)
    ax3.set_facecolor('white')
    
    # Query
    create_simple_text_box(ax3, 1, 8.5, 8, 1,
                          'Query: "Tell me about XYZ fund" (doesn\'t exist)',
                          bgcolor='#F8F9FA', textcolor='black', fontsize=12)
    
    # Baseline
    create_simple_text_box(ax3, 0.5, 6, 4, 2,
                          'GPT-4 Baseline:\nGives generic info anyway\n2,672 characters\nRisky behavior',
                          bgcolor='#FFE6E6', textcolor='black', fontsize=11)
    
    # Framework escalation
    create_simple_text_box(ax3, 5.5, 6, 4, 2,
                          'Framework:\n"Insufficient information"\nEscalated to human\nSafe behavior',
                          bgcolor='#E6F3FF', textcolor='black', fontsize=11)
    
    # Escalation highlight
    create_simple_text_box(ax3, 1.5, 3, 7, 1.5,
                          'APPROPRIATE ESCALATION\n12% of queries safely escalated',
                          bgcolor='#FFF3CD', textcolor='black', fontsize=12)
    
    ax3.axis('off')
    
    # (d) Summary Results - Clean Numbers
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.set_title('(d) Framework Achievements', fontweight='bold', fontsize=16, pad=20)
    ax4.set_facecolor('white')
    
    # Big achievement numbers
    achievements = [
        {'text': '200% Better\nFinancial Accuracy', 'pos': (0.5, 7), 'color': '#E6FFE6'},
        {'text': '82% Shorter\nResponses', 'pos': (5.5, 7), 'color': '#E6F3FF'},
        {'text': 'Zero\nHallucinations', 'pos': (0.5, 4.5), 'color': '#E6FFE6'},
        {'text': '12% Appropriate\nEscalations', 'pos': (5.5, 4.5), 'color': '#FFF3CD'},
    ]
    
    for ach in achievements:
        create_simple_text_box(ax4, ach['pos'][0], ach['pos'][1], 4, 1.5,
                              ach['text'], bgcolor=ach['color'], textcolor='black', 
                              fontsize=12, fontweight='bold')
    
    # Overall result
    create_simple_text_box(ax4, 1.5, 1.5, 7, 1.5,
                          'PRODUCTION-READY SYSTEM\nStatistically significant (p < 0.001)',
                          bgcolor='#E9ECEF', textcolor='black', fontsize=12, fontweight='bold')
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/sachin/Documents/GitHub/A-Practical-Guide-to-LLMs-Book/paper_mitigate_hallucination/initialsub/figure2_case_studies.png',
                dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

if __name__ == "__main__":
    print("Creating clean, readable figures with large text...")
    print("Generating Figure 1...")
    create_figure1_clean()
    print("Figure 1 completed!")
    
    print("Generating Figure 2...")  
    create_figure2_clean()
    print("Figure 2 completed!")
    
    print("All figures generated with large, readable text and clean layouts!")