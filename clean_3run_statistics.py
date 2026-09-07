#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu


def main():
    parser = argparse.ArgumentParser(description='Three-run clean-set statistical check for Baseline vs Consistency.')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('/root/autodl-tmp/paper_runs_20250309_final_materials/clean_summary_per_seed.csv'),
        help='CSV containing method, seed, and macro_f1 columns.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/root/autodl-tmp/clean_3run_statistics')
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {'method', 'seed', 'macro_f1'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')

    b = df.loc[df['method'].str.lower().eq('baseline'), ['seed', 'macro_f1']].dropna().copy()
    c = df.loc[df['method'].str.lower().eq('consistency'), ['seed', 'macro_f1']].dropna().copy()

    if len(b) != 3 or len(c) != 3:
        raise ValueError(f'Expected 3 runs per method, got Baseline={len(b)}, Consistency={len(c)}')

    # These runs are not seed-matched in the source CSV, so paired tests would be invalid.
    welch = ttest_ind(c['macro_f1'], b['macro_f1'], equal_var=False)
    mw = mannwhitneyu(c['macro_f1'], b['macro_f1'], alternative='two-sided', method='exact')

    out = pd.DataFrame([{
        'metric': 'Clean Macro-F1',
        'runs_per_method': 3,
        'baseline_mean': b['macro_f1'].mean(),
        'baseline_std': b['macro_f1'].std(ddof=1),
        'consistency_mean': c['macro_f1'].mean(),
        'consistency_std': c['macro_f1'].std(ddof=1),
        'mean_difference_C_minus_B': c['macro_f1'].mean() - b['macro_f1'].mean(),
        'welch_t_p': welch.pvalue,
        'mannwhitney_exact_p': mw.pvalue,
        'seed_matched': False,
    }])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.concat([
        b.assign(method='Baseline').rename(columns={'macro_f1': 'value'}),
        c.assign(method='Consistency').rename(columns={'macro_f1': 'value'})
    ], ignore_index=True)[['method', 'seed', 'value']]

    raw_path = args.output_dir / 'clean_3run_raw.csv'
    stat_path = args.output_dir / 'clean_3run_statistics.csv'
    raw.to_csv(raw_path, index=False)
    out.to_csv(stat_path, index=False)

    print(out.to_string(index=False))
    print(f'Raw: {raw_path}')
    print(f'Statistics: {stat_path}')


if __name__ == '__main__':
    main()
