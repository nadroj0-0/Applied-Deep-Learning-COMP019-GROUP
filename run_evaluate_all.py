import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from models_config import MODEL_REGISTRY

def plot_model_comparison(item_id, actual_series, all_model_preds):
    """Generates a high-level visual comparison of all models for a specific item."""
    plt.figure(figsize=(14, 7))
    
    # 1. Plot Actual Sales (The Ground Truth)
    plt.plot(actual_series.values, label='Actual Sales', color='black', linewidth=2, zorder=3)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Standard color cycle
    
    for i, (name, preds) in enumerate(all_model_preds.items()):
        item_preds = preds[preds['id'] == item_id].sort_values('day_ahead')
        color = colors[i % len(colors)]
        
        # 2. Plot the Median (q0.5)
        plt.plot(item_preds['q0.5'].values, label=f'{name} Median', color=color, linestyle='--' if 'linear' in name else '-')
        
        # 3. Plot the Uncertainty Fan (95% CI)
        # Linear models will have 0 width here, while H-LSTM will show the 'Jazz'
        plt.fill_between(
            range(28), 
            item_preds['q0.025'].values, 
            item_preds['q0.975'].values, 
            color=color, alpha=0.15, label=f'{name} 95% CI'
        )

    plt.title(f"M5 Forecast Comparison: {item_id}", fontsize=14)
    plt.xlabel("Days Ahead (1-28)", fontsize=12)
    plt.ylabel("Sales Units", fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = f"outputs/comparison_{item_id}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"📈 Visualization saved to {plot_path}")

def main():
    summary_stats = []
    all_predictions = {}
    
    # We need actual data for the plot (last 28 days of test_raw)
    # We'll grab this from the first model loaded
    actual_data = None
    top_items = None

    for name, model_class in MODEL_REGISTRY.items():
        print(f"\n--- Evaluating Model: {name} ---")
        m = model_class()
        m.load_and_split_data() #
        m.preprocess()
        
        # 1. Generate/Load Predictions
        preds_df = m.predict() #
        all_predictions[name] = preds_df
        
        # 2. Compute Metrics
        results = m.evaluate(preds_df) #
        summary_stats.append(results)
        
        # 3. Capture ground truth and top items once
        if actual_data is None:
            # target window: d_1914 to d_1941
            actual_data = m.test_raw[m.test_raw['d_num'] >= m.TARGET_START]
            # Pick top 3 items by weight to visualize
            top_items = m.item_weights.sort_values(ascending=False).head(3).index.tolist()

    # 4. Generate Visualizations for Top Items
    print("\n--- Generating Model Visualizations ---")
    for item_id in top_items:
        item_actuals = actual_data[actual_data['id'] == item_id].sort_values('d_num')['sales']
        plot_model_comparison(item_id, item_actuals, all_predictions)

    # 5. Create Comparison Table
    df_compare = pd.DataFrame(summary_stats)
    print("\n" + "="*50)
    print("      FINAL MODEL COMPARISON (M5 STANDARDS)")
    print("="*50)
    print(df_compare.to_string(index=False))
    
    df_compare.to_csv("outputs/model_comparison.csv", index=False)

if __name__ == "__main__":
    main()