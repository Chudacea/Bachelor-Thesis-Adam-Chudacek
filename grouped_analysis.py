import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

# --- GLOBÁLNÍ NASTAVENÍ FONTŮ ---
"""plt.rcParams.update({
    'font.size': 16,               
    'axes.titlesize': 22,          
    'axes.labelsize': 18,          
    'xtick.labelsize': 17,         
    'ytick.labelsize': 20,         
    'legend.fontsize': 17,         
    'figure.titlesize': 20         
})"""
# ---------------------------------------------

# ==========================================
# SHARED CONFIGURATION & FUNCTIONS
# ==========================================
HIGHLIGHT_AA = "W"
AA_ORDER = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

def get_aa_col_index(aa):
    return 3 + AA_ORDER.index(aa)

def parse_state_file(filepath, nodes_of_interest):
    node_data = {node: [] for node in nodes_of_interest}
    with open(filepath, "r") as f:
        lines = f.readlines()[9:] 
        for line in lines:
            parts = line.split()
            if parts and parts[0] in node_data:
                node_data[parts[0]].append(parts)
    return node_data

def get_asr_quality(nodes, node_data):
    qualities = []
    for n in nodes:
        node_max_probs = [max(float(x) for x in p[3:]) for p in node_data[n]]
        qualities.append(sum(node_max_probs) / len(node_max_probs) if node_max_probs else 0)
    return qualities

def get_metrics(nodes, node_data, target_aa):
    col_idx = get_aa_col_index(target_aa)
    exp = [sum(float(p[col_idx]) for p in node_data[n]) for n in nodes]
    avg = [sum(float(p[col_idx]) for p in node_data[n]) / len(node_data[n]) if node_data[n] else 0.0 for n in nodes]
    counts = [sum(1 for p in node_data[n] if p[2] == target_aa) for n in nodes]
    
    log_probs = []
    for n in nodes:
        log_none = 0.0
        has_certain = False
        for p in node_data[n]:
            prob_tgt = float(p[col_idx])
            if prob_tgt >= 1.0:
                has_certain = True
                break
            if 1.0 - prob_tgt > 0:
                log_none += math.log10(1.0 - prob_tgt)
        log_probs.append(-100 if has_certain else log_none)
        
    return exp, avg, counts, log_probs

# ==========================================
# PART 1: GLUCOKINASE (GLK) ANALYSIS
# ==========================================
print("--- Starting GLK Analysis ---")

input_glk = "/Users/adamchudacek/Documents/Marburg/Bakalarka/Vysledky/populated_glucokinase/work/ezASR_results/AA_trimmed.state"
output_glk = "/Users/adamchudacek/Documents/Marburg/Bakalarka/Vysledky/grouped/glk"
os.makedirs(output_glk, exist_ok=True)

path1_nodes = ["Anc_42", "Anc_23", "Anc_17", "Anc_13", "Anc_12", "Anc_11", "Anc_1"]
path2_nodes = ["Anc_16", "Anc_15", "Anc_14", "Anc_13", "Anc_12", "Anc_11", "Anc_1"]

x_labels_glk = [n1 if n1 == n2 else f"{n1}\n/\n{n2}" for n1, n2 in zip(path1_nodes, path2_nodes)]
x_glk = range(len(x_labels_glk))
PI_TRP_GLK = 0.012

all_glk_nodes = set(path1_nodes + path2_nodes)
node_data_glk = parse_state_file(input_glk, all_glk_nodes)

trp_exp_p1, trp_avg_p1, trp_counts_p1, trp_log_p1 = get_metrics(path1_nodes, node_data_glk, HIGHLIGHT_AA)
trp_exp_p2, trp_avg_p2, trp_counts_p2, trp_log_p2 = get_metrics(path2_nodes, node_data_glk, HIGHLIGHT_AA)
asr_q_p1 = get_asr_quality(path1_nodes, node_data_glk)
asr_q_p2 = get_asr_quality(path2_nodes, node_data_glk)

# --- GLK 4-PANEL DASHBOARD ---
fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)

axes[0].plot(x_glk, asr_q_p1, marker='s', color='black', linestyle='-', linewidth=2.5, markersize=8, label="Path 1")
axes[0].plot(x_glk, asr_q_p2, marker='o', color='gray', linestyle='--', linewidth=2.5, markersize=8, label="Path 2")
axes[0].set_title("A. Ancestral Sequence Reconstruction Quality", loc='left', fontweight='bold')
axes[0].set_ylabel("Mean Max Posterior Prob")
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(loc="upper right")

axes[1].plot(x_glk, trp_counts_p1, marker='s', color='purple', linestyle='-', linewidth=2.5, markersize=8, label="Path 1")
axes[1].plot(x_glk, trp_counts_p2, marker='o', color='purple', linestyle='--', linewidth=2.5, markersize=8, label="Path 2")
axes[1].set_title("B. Top-Ranked Position Counts for Tryptophan (W)", loc='left', fontweight='bold')
axes[1].set_ylabel("Number of Positions")
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(loc="upper right")

axes[2].plot(x_glk, trp_exp_p1, marker='s', color='purple', linestyle='-', linewidth=2.5, markersize=8, label="Path 1")
axes[2].plot(x_glk, trp_exp_p2, marker='o', color='purple', linestyle='--', linewidth=2.5, markersize=8, label="Path 2")
axes[2].set_title("C. Expectation Value for Tryptophan (W)", loc='left', fontweight='bold')
axes[2].set_ylabel("Expected Residue Count")
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].legend(loc="upper right")

axes[3].plot(x_glk, trp_log_p1, marker='s', color='purple', linestyle='-', linewidth=2.5, markersize=8, label="Path 1")
axes[3].plot(x_glk, trp_log_p2, marker='o', color='purple', linestyle='--', linewidth=2.5, markersize=8, label="Path 2")
axes[3].set_title("D. Log10 Probability of Absence for Tryptophan (W)", loc='left', fontweight='bold')
axes[3].set_ylabel("Log10 Probability")
axes[3].grid(True, linestyle='--', alpha=0.6)
axes[3].legend(loc="upper right")

axes[3].set_xticks(x_glk)
axes[3].set_xticklabels(x_labels_glk)
axes[3].set_xlabel("Ancestral Nodes")

plt.tight_layout()
plt.savefig(os.path.join(output_glk, "GLK_4Panel_Dashboard.png"), dpi=300)
plt.close()

# --- GLK Evolution Signal ---
plt.figure(figsize=(12, 6))
plt.plot(x_glk, trp_avg_p1, marker='s', color='purple', linestyle='-', linewidth=2.5, markersize=8, label="Path 1")
plt.plot(x_glk, trp_avg_p2, marker='o', color='purple', linestyle='--', linewidth=2.5, markersize=8, label="Path 2")
plt.axhline(y=PI_TRP_GLK, color='red', linestyle=':', linewidth=2, label=rf'Baseline $\pi_W$ ({PI_TRP_GLK*100:.1f}%)')
plt.title("Evolution Signal: Mean Posterior Probability of Tryptophan", pad=12, fontweight='bold')
plt.ylabel("Mean Posterior Probability")
plt.xticks(x_glk, x_labels_glk)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_glk, "GLK_Evolution_Signal.png"), dpi=300)
plt.close()

# ==========================================
# PART 2: ALDOLASE (FBAA) ANALYSIS
# ==========================================
print("\n--- Starting FBAA Analysis ---")

input_fbaa = "/Users/adamchudacek/Documents/Marburg/Bakalarka/Vysledky/Imag_BAKA/fbaA/ezASR_results/AA_trimmed.state"
output_fbaa = "/Users/adamchudacek/Documents/Marburg/Bakalarka/Vysledky/grouped/fbaA"
os.makedirs(output_fbaa, exist_ok=True)

pathA_nodes = [
    "Anc_74", "Anc_73", "Anc_72", "Anc_71", "Anc_70", "Anc_69", "Anc_68", "Anc_67", 
    "Anc_61", "Anc_50", "Anc_49", "Anc_48", "Anc_37", "Anc_36", "Anc_35", "Anc_34", 
    "Anc_33", "Anc_32", "Anc_25", "Anc_14", "Anc_13", "Anc_12", "Anc_11", "Anc_10"
]
PI_TRP_FBAA = 0.029
x_fbaa = range(len(pathA_nodes))

node_data_fbaa = parse_state_file(input_fbaa, set(pathA_nodes))

asr_quality_fbaa = get_asr_quality(pathA_nodes, node_data_fbaa)
trp_exp_fbaa, trp_avg_fbaa, trp_counts_fbaa, trp_log_fbaa = get_metrics(pathA_nodes, node_data_fbaa, HIGHLIGHT_AA)

# --- FBAA 4-PANEL DASHBOARD ---
fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)

axes[0].plot(x_fbaa, asr_quality_fbaa, marker='D', color='black', linewidth=2.5, markersize=8, label="ASR Quality")
axes[0].set_title("A. Ancestral Sequence Reconstruction Quality", loc='left', fontweight='bold')
axes[0].set_ylabel("Mean Max Posterior Prob")
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(loc="upper right")

axes[1].plot(x_fbaa, trp_counts_fbaa, marker='o', color='purple', linewidth=2.5, markersize=8, label="W (Trp)")
axes[1].set_title("B. Top-Ranked Position Counts", loc='left', fontweight='bold')
axes[1].set_ylabel("Number of Positions")
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(loc="upper right")

axes[2].plot(x_fbaa, trp_exp_fbaa, marker='o', color='purple', linewidth=2.5, markersize=8, label="W (Trp)")
axes[2].set_title("C. Expectation Value", loc='left', fontweight='bold')
axes[2].set_ylabel("Expected Residue Count")
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].legend(loc="upper right")

axes[3].plot(x_fbaa, trp_avg_fbaa, marker='o', color='purple', linewidth=2.5, markersize=8, label="W (Trp)")
axes[3].axhline(y=PI_TRP_FBAA, color='red', linestyle='--', linewidth=2, label=rf'Baseline $\pi_W$ ({PI_TRP_FBAA*100:.1f}%)')
axes[3].set_title("D. Evolution Signal (Mean Posterior Prob vs Baseline)", loc='left', fontweight='bold')
axes[3].set_ylabel("Mean Posterior Probability")
axes[3].grid(True, linestyle='--', alpha=0.6)
axes[3].legend(loc="upper right")

axes[3].set_xticks(x_fbaa)
axes[3].set_xticklabels(pathA_nodes, rotation=45, ha='right', fontsize=17) 
axes[3].set_xlabel("Ancestral Nodes")

plt.tight_layout()
plt.savefig(os.path.join(output_fbaa, "FBAA_4Panel_Dashboard.png"), dpi=300)
plt.close()

# ==========================================
# PART 3: CORRELATIONS & COMPARISONS
# ==========================================
print("\n--- Generating Plots & Comparisons ---")

output_combined = "/Users/adamchudacek/Documents/Marburg/Bakalarka/Vysledky/grouped"
os.makedirs(output_combined, exist_ok=True)

# ------------------------------------------
# 1. POOLED CORRELATION (GLK + FBAA)
# ------------------------------------------
def plot_pooled_correlation(x_data, y_data, xlabel, ylabel, title, filename, output_dir):
    plt.figure(figsize=(10, 8))
    r, p_value = pearsonr(x_data, y_data)
    sns.regplot(
        x=x_data, y=y_data, 
        scatter_kws={'s': 120, 'color': 'purple', 'edgecolor': 'black', 'alpha': 0.7},
        line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 2.5}
    )
    text_str = f"Pooled Data (GLK + fbaA)\nPearson $r$ = {r:.3f}\n$p$-value = {p_value:.3e}\n$R^2$ = {r**2:.3f}"
    plt.text(
        0.05, 0.95, text_str, transform=plt.gca().transAxes, 
        fontsize=17, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    )
    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

print("-> Generating Pooled N_win vs Expectation Value Correlation")
pooled_counts = trp_counts_p1 + trp_counts_fbaa
pooled_exp = trp_exp_p1 + trp_exp_fbaa

plot_pooled_correlation(
    x_data=pooled_counts, 
    y_data=pooled_exp, 
    xlabel="Top-Ranked Position Counts ($N_{win}$)", 
    ylabel="Expectation Value for Tryptophan", 
    title="Correlation: Top-Ranked Counts vs. Expectation Value",
    filename="Pooled_Correlation_Nwin_vs_Exp.png",
    output_dir=output_combined
)

# ------------------------------------------
# 2. SINGLE CORRELATIONS (FBAA od Anc_50)
# ------------------------------------------
def plot_single_correlation(x_data, y_data, xlabel, ylabel, title, filename, output_dir):
    plt.figure(figsize=(10, 8))
    r, p_value = pearsonr(x_data, y_data)
    sns.regplot(
        x=x_data, y=y_data, 
        scatter_kws={'s': 120, 'color': 'teal', 'edgecolor': 'black', 'alpha': 0.8},
        line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 2.5}
    )
    text_str = f"Pearson $r$ = {r:.3f}\n$p$-value = {p_value:.3e}\n$R^2$ = {r**2:.3f}"
    plt.text(
        0.05, 0.95, text_str, transform=plt.gca().transAxes, 
        fontsize=17, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    )
    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

print("-> Generating FBAA ASR Quality Correlation")
start_idx = pathA_nodes.index("Anc_50")
asr_quality_fbaa_sliced = asr_quality_fbaa[start_idx:]
trp_exp_fbaa_sliced = trp_exp_fbaa[start_idx:]
trp_counts_fbaa_sliced = trp_counts_fbaa[start_idx:]

plot_single_correlation(
    x_data=asr_quality_fbaa_sliced, 
    y_data=trp_exp_fbaa_sliced, 
    xlabel="ASR Quality (Mean Max Posterior Prob)", 
    ylabel="Expectation Value for Tryptophan", 
    title="Correlation: ASR Quality vs. Exp. Value (from Anc_50)",
    filename="FBAA_Correlation_Quality_vs_Exp_from_Anc50.png",
    output_dir=output_fbaa
)

print("-> Generating FBAA N_win vs Exp Correlation (from Anc_50)")
plot_single_correlation(
    x_data=trp_counts_fbaa_sliced, 
    y_data=trp_exp_fbaa_sliced, 
    xlabel="Top-Ranked Position Counts ($N_{win}$)", 
    ylabel="Expectation Value for Tryptophan", 
    title="Correlation: N_win vs. Exp. Value (from Anc_50)",
    filename="FBAA_Correlation_Nwin_vs_Exp_from_Anc50.png",
    output_dir=output_fbaa
)

# ------------------------------------------
# GLK GROUP ABUNDANCE: EARLY VS LATE (Path 1 vs Path 2)
# ------------------------------------------
print("-> Generating GLK Group Abundance (Early vs Late) Comparison")

AA_GROUP_1 = ["G", "S", "Q", "N", "P", "I", "L", "V", "A", "T"]
GROUP_1_NAME = "Early AAs"

AA_GROUP_2 = ["R", "K", "H", "D", "E", "C", "M", "F", "Y", "W"]
GROUP_2_NAME = "Late AAs"

def get_group_exp_sum(path_nodes, node_data, aa_group):
    total_exp = np.zeros(len(path_nodes))
    for aa in aa_group:
        exp, _, _, _ = get_metrics(path_nodes, node_data, aa)
        total_exp += np.array(exp)
    return total_exp.tolist()

group1_p1 = get_group_exp_sum(path1_nodes, node_data_glk, AA_GROUP_1)
group1_p2 = get_group_exp_sum(path2_nodes, node_data_glk, AA_GROUP_1)

group2_p1 = get_group_exp_sum(path1_nodes, node_data_glk, AA_GROUP_2)
group2_p2 = get_group_exp_sum(path2_nodes, node_data_glk, AA_GROUP_2)

# Vykreslení přesně podle vizuálu ze Screenshot 2026-08-15 at 20.20.49.jpg
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

# --- Horní panel: Early AAs ---
ax_top.plot(x_glk, group1_p1, marker='o', linewidth=2.5, color='teal', linestyle='-', label=f"Path 1")
ax_top.plot(x_glk, group1_p2, marker='o', linewidth=2.5, color='teal', linestyle='--', label=f"Path 2")
ax_top.set_title(f"Amino Acid Group Comparison: {GROUP_1_NAME} vs. {GROUP_2_NAME}", fontsize=14, pad=12)
ax_top.set_ylabel(f"Expected Count ({GROUP_1_NAME})", fontsize=11)
ax_top.grid(True, linestyle='--', alpha=0.6)
ax_top.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

# --- Spodní panel: Late AAs ---
ax_bot.plot(x_glk, group2_p1, marker='s', linewidth=2.5, color='darkorange', linestyle='-', label=f"Path 1")
ax_bot.plot(x_glk, group2_p2, marker='s', linewidth=2.5, color='darkorange', linestyle='--', label=f"Path 2")
ax_bot.set_xlabel("Ancestral Nodes", fontsize=11)
ax_bot.set_ylabel(f"Expected Count ({GROUP_2_NAME})", fontsize=11)
ax_bot.set_xticks(x_glk)
ax_bot.set_xticklabels(x_labels_glk, rotation=45, ha='right')
ax_bot.grid(True, linestyle='--', alpha=0.6)
ax_bot.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
# Parametr bbox_inches='tight' zabrání oříznutí legendy, která je mimo graf
plt.savefig(os.path.join(output_glk, "GLK_Group_Abundance_Path1_vs_Path2.png"), dpi=300, bbox_inches='tight')
plt.close()