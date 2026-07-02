import pandas as pd
import numpy as np
import math
import os

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'semantic_similarity_classification.csv')
        df = pd.read_csv(csv_path)
        scores = df['semantic_similarity'].tolist()
        
        mean = np.mean(scores)
        std = np.std(scores)
        n = len(scores)

        ci = 1.96 * (std / math.sqrt(n))

        # Calculate IQR-based threshold (matching human-like scorer approach)
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        iqr_threshold = q1 - 1.5 * iqr

        print(f"Total samples: {n}")
        print(f"\nDescriptive Statistics:")
        print(f"Mean: {mean}")
        print(f"Std dev: {std}")
        print(f"95% Confidence Interval: ({mean - ci}, {mean + ci})")

        print(f"\nPercentiles:")
        percentile_1 = np.percentile(scores, 1)
        percentile_5 = np.percentile(scores, 5)
        print(f"1st percentile: {percentile_1}")
        print(f"5th percentile: {percentile_5}")
        print(f"Q1 (25th percentile): {q1}")
        print(f"Median (50th percentile): {np.percentile(scores, 50)}")
        print(f"Q3 (75th percentile): {q3}")

        print(f"\nIQR-based Threshold:")
        print(f"IQR: {iqr}")
        print(f"Threshold (Q1 - 1.5×IQR): {iqr_threshold}")

        # Check if edit_type column exists for breakdown
        if 'edit_type' in df.columns:
            print(f"\nBreakdown by Edit Type:")
            for edit_type in sorted(df['edit_type'].unique()):
                type_scores = df[df['edit_type'] == edit_type]['semantic_similarity'].tolist()
                type_mean = np.mean(type_scores)
                type_q1 = np.percentile(type_scores, 25)
                type_q3 = np.percentile(type_scores, 75)
                type_iqr = type_q3 - type_q1
                type_threshold = type_q1 - 1.5 * type_iqr
                print(f"\n  {edit_type} (n={len(type_scores)}):")
                print(f"    Mean: {type_mean:.4f}")
                print(f"    Q1: {type_q1:.4f}, Q3: {type_q3:.4f}, IQR: {type_iqr:.4f}")
                print(f"    IQR-based threshold: {type_threshold:.4f}")

    except FileNotFoundError:
        print("Error: The file semantic_similarity_classification.csv was not found.")
    except KeyError:
        print("Error: The CSV file must have a column named 'semantic_similarity'.")

if __name__ == '__main__':
    main()
