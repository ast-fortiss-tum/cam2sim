import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import math

# -------------- CONFIGURAZIONE VISUALIZZAZIONE PANDAS --------------
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 100)

# -------------- CONFIGURAZIONE METRICHE --------------

HIGHER_IS_BETTER = {
    "PSNR", "SSIM", "SegScore",
    "Veh_Recall", "Veh_Precision", "Veh_AvgIoU",
    "IS_mean", "PRDC_Precision", "PRDC_Recall",
    "PRDC_Density", "PRDC_Coverage", "Precision", "Recall", "Density", "Coverage",
    # Temporal: higher SSIM/PSNR = more consistent between frames
    "Temp_SSIM", "Temp_PSNR",
}

LOWER_IS_BETTER = {
    "MSE", "FID", "KID_mean", "MMD_RBF", "CPL",
    # Temporal: lower MSE/CPL = more consistent between frames
    "Temp_MSE", "Temp_CPL",
}

METRIC_GROUPS = {
    "Score_Vehicle": {
        "Veh_Recall", "Veh_Precision", "Veh_AvgIoU"
    },
    "Score_Distribution": {
        "FID", "KID_mean", "IS_mean", "MMD_RBF",
        "PRDC_Precision", "PRDC_Recall", "PRDC_Density", "PRDC_Coverage",
        "Precision", "Recall", "Density", "Coverage"
    },
    "Score_SingleImage": {
        "PSNR", "SSIM", "MSE", "SegScore", "CPL"
    },
    "Score_Temporal": {
        "Temp_SSIM", "Temp_PSNR", "Temp_MSE", "Temp_CPL"
    },
}

EXCLUDE_FROM_SCORING = {"Start_Seg", "Start_Inst", "Start_Temp", "End_Seg", "End_Inst", "End_Temp"}


# -------------- FUNZIONI UTILITY --------------

def parse_filename_params(name):
    params = {
        "Start_Seg": 0.0, "Start_Inst": 0.0, "Start_Temp": 0.0,
        "End_Seg":   0.0, "End_Inst":   0.0, "End_Temp":   0.0,
    }
    m = re.search(
        r"START_seg_([\d\.]+)_inst_([\d\.]+)_temp_([\d\.]+)__END_seg_([\d\.]+)_inst_([\d\.]+)_temp_([\d\.]+)",
        name
    )
    if m:
        params["Start_Seg"]  = float(m.group(1))
        params["Start_Inst"] = float(m.group(2))
        params["Start_Temp"] = float(m.group(3))
        params["End_Seg"]    = float(m.group(4))
        params["End_Inst"]   = float(m.group(5))
        params["End_Temp"]   = float(m.group(6))
    return params


def looks_like_metrics_dict(d):
    if not isinstance(d, dict):
        return False
    metric_markers = {"FID", "IS_mean", "PSNR", "SSIM", "MSE", "KID_mean", "Veh_Recall", "Veh_AvgIoU", "Temp_SSIM"}
    return any(k in d for k in metric_markers)


# -------------- HUMAN RANKINGS --------------

def load_human_rankings(human_csv):
    df = pd.read_csv(human_csv)
    if 'config' not in df.columns or 'human_score' not in df.columns:
        raise ValueError(f"Human CSV needs 'config' and 'human_score'. Found: {df.columns.tolist()}")
    return df[['rank', 'config', 'human_score']].sort_values('rank').reset_index(drop=True)


def match_human_to_metrics(human_df, metric_index):
    mapping = {}
    for _, row in human_df.iterrows():
        hconfig = row['config']
        matched = False
        for sys_name in metric_index:
            clean_sys = re.sub(r'_image_level_report$', '', sys_name)
            if clean_sys == hconfig or sys_name == hconfig:
                mapping[sys_name] = {'rank': row['rank'], 'human_score': row['human_score']}
                matched = True
                break
        if not matched:
            for sys_name in metric_index:
                if hconfig in sys_name or sys_name in hconfig:
                    mapping[sys_name] = {'rank': row['rank'], 'human_score': row['human_score']}
                    break
    return mapping


def compute_correlations(df_scored, human_mapping):
    from scipy.stats import spearmanr, kendalltau
    matched = [s for s in df_scored.index if s in human_mapping]
    if len(matched) < 4:
        print(f"  Only {len(matched)} matched - too few for correlation")
        return None
    human_scores = pd.Series({s: human_mapping[s]['human_score'] for s in matched})
    df_sub = df_scored.loc[matched]
    numeric_cols = [c for c in df_sub.select_dtypes(include=["number"]).columns
                    if c not in EXCLUDE_FROM_SCORING
                    and c not in ('human_score', 'human_rank')
                    and (df_sub[c] >= 0).any()]
    rows = []
    for col in sorted(numeric_cols):
        valid = df_sub[col].notna() & (df_sub[col] != float('inf'))
        if valid.sum() < 4:
            continue
        rho, p_rho = spearmanr(human_scores[valid], df_sub.loc[valid, col])
        tau, p_tau = kendalltau(human_scores[valid], df_sub.loc[valid, col])
        rows.append({
            'metric': col, 'spearman_rho': round(rho, 4), 'spearman_p': round(p_rho, 4),
            'kendall_tau': round(tau, 4), 'kendall_p': round(p_tau, 4),
            'abs_rho': round(abs(rho), 4), 'n': int(valid.sum()),
            'sig': '*' if p_rho < 0.05 else '',
        })
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values('abs_rho', ascending=False)


def print_correlations(corr_df):
    print("\n" + "=" * 90)
    print("  HUMAN vs METRIC CORRELATION (Spearman rho)")
    print("=" * 90)
    print("  |rho| > 0.7 = strong    0.4-0.7 = moderate    < 0.4 = weak")
    print("  Negative rho for lower-is-better metrics (FID, MSE...) = CORRECT\n")
    print(f"  {'Metric':<24} {'Spearman':>9} {'p-val':>8} {'Kendall':>9} {'p-val':>8} {'Sig':>4} {'N':>3}")
    print("  " + "-" * 72)
    for _, r in corr_df.iterrows():
        a = abs(r['spearman_rho'])
        icon = "G" if a >= 0.7 else "M" if a >= 0.4 else "W"
        print(f"  [{icon}] {r['metric']:<22} {r['spearman_rho']:>7.4f} {r['spearman_p']:>8.4f}"
              f"   {r['kendall_tau']:>7.4f} {r['kendall_p']:>8.4f}  {r['sig']:>3} {r['n']:>3}")
    print("  " + "-" * 72)
    best = corr_df.iloc[0]
    print(f"\n  Best correlator: {best['metric']} (rho={best['spearman_rho']:.4f}, p={best['spearman_p']:.4f})")


# -------------- CORE SCRIPT --------------

def load_reports(folder, include_params=True):
    systems = {}
    if not os.path.exists(folder):
        print(f"Errore: Cartella '{folder}' non trovata.")
        return {}
    print(f"Analisi cartella: {folder}")
    count_loaded = 0
    count_failed = 0
    count_skipped = 0
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".json"):
            continue
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Errore lettura {fpath}: {e}")
            count_failed += 1
            continue
        system_name = os.path.splitext(fname)[0]
        avg_metrics = data.get("average_metrics")
        if avg_metrics is None:
            if looks_like_metrics_dict(data):
                avg_metrics = data
            else:
                count_skipped += 1
                continue

        # Merge temporal metrics into avg_metrics if present
        temporal_metrics = data.get("temporal_metrics")
        if temporal_metrics and isinstance(temporal_metrics, dict):
            avg_metrics = dict(avg_metrics)
            avg_metrics.update(temporal_metrics)

        if include_params:
            avg_metrics = dict(avg_metrics)
            avg_metrics.update(parse_filename_params(fname))
        systems[system_name] = avg_metrics
        count_loaded += 1
    print(f"Caricati: {count_loaded}  Saltati: {count_skipped}  Falliti: {count_failed}")
    return systems


def build_dataframe(systems_dict):
    if not systems_dict:
        raise ValueError("Nessun sistema caricato.")
    df = pd.DataFrame.from_dict(systems_dict, orient="index")
    df.index.name = "system"
    return df


def compute_scores(df):
    df_numeric = df.select_dtypes(include=["number"]).copy()
    df_numeric = df_numeric.drop(columns=[c for c in df_numeric.columns if c in EXCLUDE_FROM_SCORING], errors="ignore")
    if df_numeric.empty:
        raise ValueError("Nessuna metrica numerica trovata.")
    means = df_numeric.mean()
    stds = df_numeric.std(ddof=0)
    stds_safe = stds.replace(0, 1)
    z_df = (df_numeric - means) / stds_safe
    z_df = z_df.fillna(0)
    for col in df_numeric.columns:
        if col in LOWER_IS_BETTER:
            z_df[col] = -z_df[col]
    results = df.copy()
    for group_name, metric_set in METRIC_GROUPS.items():
        valid_cols = [c for c in z_df.columns if c in metric_set]
        results[group_name] = z_df[valid_cols].mean(axis=1) if valid_cols else np.nan
    results["Score_Overall"] = z_df.mean(axis=1)
    return results


def main(folder, show_plot=True, save_csv=None, include_params=True, human_csv=None):
    # 1. Load
    systems = load_reports(folder, include_params=include_params)
    if not systems:
        print("Nessun file valido trovato.")
        return

    # 2. DataFrame + Scores
    df = build_dataframe(systems)
    try:
        df_scored = compute_scores(df)
    except Exception as e:
        print(f"Errore: {e}")
        return

    # 3. Human rankings (if provided)
    human_mapping = {}
    human_order = None
    corr_df = None

    if human_csv:
        try:
            human_df = load_human_rankings(human_csv)
            print(f"\nHuman rankings loaded: {len(human_df)} configs")
            human_mapping = match_human_to_metrics(human_df, df_scored.index)
            print(f"Matched {len(human_mapping)} / {len(human_df)} to metric data")

            df_scored['human_rank'] = df_scored.index.map(
                lambda s: human_mapping[s]['rank'] if s in human_mapping else np.nan)
            df_scored['human_score'] = df_scored.index.map(
                lambda s: human_mapping[s]['human_score'] if s in human_mapping else np.nan)

            matched = [s for s in df_scored.index if s in human_mapping]
            human_order = sorted(matched, key=lambda s: human_mapping[s]['rank'])

            corr_df = compute_correlations(df_scored, human_mapping)
            if corr_df is not None:
                print_correlations(corr_df)
        except Exception as e:
            print(f"Error loading human rankings: {e}")
            human_csv = None

    # 4. Sort always by Score_Overall, human info shown as columns
    if "Score_Overall" in df_scored.columns:
        df_sorted = df_scored.sort_values("Score_Overall", ascending=False)
        sort_label = "Score_Overall" + (" (with human ranking)" if human_csv else "")
    else:
        df_sorted = df_scored
        sort_label = "unsorted"

    # 5. Print
    print("\n" + "=" * 80)
    print(f"CLASSIFICA (sorted by: {sort_label})")
    print("=" * 80)
    display_cols = [c for c in ["human_rank", "human_score", "Score_Overall",
                                 "Score_SingleImage", "Score_Distribution", "Score_Vehicle",
                                 "Score_Temporal"]
                    if c in df_sorted.columns]
    print(df_sorted[display_cols].head(10))

    if "Score_Vehicle" in df_scored.columns:
        print("\n" + "=" * 80)
        print("TOP 5 DETECTION VEICOLI (Score_Vehicle)")
        print("=" * 80)
        df_vehicle = df_scored.sort_values("Score_Vehicle", ascending=False)
        cols_veh = ["Score_Vehicle"] + [c for c in ["Veh_AvgIoU", "Veh_Recall", "Veh_Precision"] if c in df_scored.columns]
        print(df_vehicle[cols_veh].head(5))

    if "Score_Temporal" in df_scored.columns:
        print("\n" + "=" * 80)
        print("TOP 5 TEMPORAL CONSISTENCY (Score_Temporal)")
        print("=" * 80)
        df_temporal = df_scored.sort_values("Score_Temporal", ascending=False)
        cols_temp = ["Score_Temporal"] + [c for c in ["Temp_SSIM", "Temp_PSNR", "Temp_MSE", "Temp_CPL"] if c in df_scored.columns]
        print(df_temporal[cols_temp].head(5))

    # 6. Save
    if save_csv:
        top_n = 25
        top_configs = df_sorted.index[:top_n].tolist()
        top_configs_clean = [re.sub(r'_image_level_report$', '', c) for c in top_configs]

        txt_path = save_csv.replace('.csv', f'_top{top_n}.txt') if save_csv else f'top{top_n}_configs.txt'
        with open(txt_path, 'w') as f:
            for i, name in enumerate(top_configs_clean, 1):
                f.write(f"{name}\n")
        print(f"Top {top_n} configs saved: {txt_path}")
        print(f"\nTop {min(top_n, len(top_configs_clean))} configs:")
        for i, name in enumerate(top_configs_clean, 1):
            print(f"  {i:>2}. {name}")

        df_sorted.to_csv(save_csv)
        print(f"\nCSV salvato: {save_csv}")
    if corr_df is not None:
        corr_path = (save_csv.replace('.csv', '_correlation.csv') if save_csv else 'correlation_results.csv')
        corr_df.to_csv(corr_path, index=False)
        print(f"Correlation CSV: {corr_path}")

    # 7. Plot
    if show_plot:
        plot_cols = ["Score_SingleImage", "Score_Distribution", "Score_Vehicle", "Score_Temporal"]
        plot_cols = [c for c in plot_cols if c in df_sorted.columns and not df_sorted[c].isna().all()]
        if not plot_cols:
            return

        # Generate A, B, C, ... labels
        alpha_labels = [chr(65 + i) for i in range(len(df_sorted))]
        
        # Explicit color mapping for consistency across all scripts
        COLOR_MAP = {
            "Score_SingleImage": "#1f77b4",   # blue
            "Score_Distribution": "#9467bd",  # purple
            "Score_Vehicle": "#2ca02c",       # green
            "Score_Temporal": "#ff7f0e",      # orange
        }
        plot_colors = [COLOR_MAP.get(c, '#333333') for c in plot_cols]

        width = max(14, len(df_sorted) * 0.9)
        fig, ax = plt.subplots(figsize=(width, 10))
        df_sorted[plot_cols].plot(kind="bar", ax=ax, width=0.7, color=plot_colors, alpha=0.85)
        ax.set_xticklabels(alpha_labels, rotation=0, ha='center', fontsize=14, fontweight='bold')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

        plt.title("Model Comparison: Normalized Scores (Z-Score)", fontsize=18)
        
        # Show human rank overlay if available (subtract 1 to exclude GT)
        if human_mapping:
            for i, sys_name in enumerate(df_sorted.index):
                if sys_name in human_mapping:
                    hrank = human_mapping[sys_name]['rank']
                    ax.text(i, ax.get_ylim()[1] * 0.95, f"#{hrank - 1}",
                            ha='center', va='top', fontsize=13, fontweight='bold',
                            color='red', alpha=0.9)

        plt.ylabel("Standardized Score (Higher is better)", fontsize=15)
        plt.xlabel("Configuration", fontsize=15)
        ax.legend(title="Metric Category", loc='lower left', fontsize=12, title_fontsize=13)
        ax.grid(axis='y', alpha=0.2)
        ax.tick_params(axis='y', labelsize=12)

        plt.tight_layout()
        plt.savefig('metrics_plot.png', dpi=400, bbox_inches='tight', facecolor='white')
        print(f"Plot saved: metrics_plot.png")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Cartella JSON con i report metriche")
    parser.add_argument("--no-plot", action="store_true", help="Disable matplotlib plot")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--no-params", action="store_true",
                        help="Do not parse START/END params from filename into columns")
    parser.add_argument("--human", type=str, default=None,
                        help="Path to human_rankings.csv - reorders plot by human ranking "
                             "and computes Spearman correlation vs each metric")
    args = parser.parse_args()
    main(
        args.folder,
        show_plot=not args.no_plot,
        save_csv=args.csv,
        include_params=not args.no_params,
        human_csv=args.human,
    )