import math
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ==========================================
# CONFIGURATION & HEADER
# ==========================================
input_file = ""
output_folder = ""

nodes_of_interest = ["Anc_74", "Anc_73", "Anc_72","Anc_71", "Anc_70", "Anc_69", "Anc_68", "Anc_67", "Anc_61", "Anc_50", "Anc_49", "Anc_48", "Anc_37", "Anc_36", "Anc_35", "Anc_34", "Anc_33", "Anc_32", "Anc_25", "Anc_14", "Anc_13", "Anc_12", "Anc_11", "Anc_10"]
HIGHLIGHT_AA = "W"  # Target AA for solo plots

# --- DEFINICE SKUPIN AMINOKYSELIN ---
AA_GROUP_1 = ["G", "S", "Q", "N", "P", "I", "L", "V", "A", "T"]
GROUP_1_NAME = "Early AAs"

AA_GROUP_2 = ["R", "K", "H", "D", "E", "C", "M", "F", "Y", "W"]
GROUP_2_NAME = "Late AAs"

# Stationary frequency for Tryptophan baseline
PI_TRP = 0.029  

AA_ORDER = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

def get_aa_col_index(aa):
    if aa not in AA_ORDER:
        raise ValueError(f"Unknown Amino Acid: {aa}")
    return 3 + AA_ORDER.index(aa)

# ==========================================
# STYLING (Unikátní markery pro celou abecedu)
# ==========================================
markers_list = ['o', 's', '^', 'v', 'D', 'p', 'P', '*', 'X', 'h']

# ==========================================
# 1. DATA LOADING
# ==========================================
print("Loading and parsing file...")
node_data = {node: [] for node in nodes_of_interest}

with open(input_file, "r") as f:
    lines = f.readlines()[9:]
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        node = parts[0]
        if node in node_data:
            node_data[node].append(parts)

print("Data loaded. Calculating parameters...")

# ==========================================
# 2. CALCULATION FUNCTIONS
# ==========================================
def get_expectation_values(nodes, node_data, target_aa):
    col_idx = get_aa_col_index(target_aa)
    return [sum(float(parts[col_idx]) for parts in node_data[node]) for node in nodes]

def get_average_probabilities(nodes, node_data, target_aa):
    col_idx = get_aa_col_index(target_aa)
    avg_probs = []
    for node in nodes:
        sites = node_data[node]
        avg_probs.append(sum(float(parts[col_idx]) for parts in sites) / len(sites) if sites else 0.0)
    return avg_probs

def get_certainty_counts(nodes, node_data, target_aa):
    return [sum(1 for parts in node_data[node] if parts[2] == target_aa) for node in nodes]

def get_asr_quality(nodes, node_data):
    qualities = []
    for node in nodes:
        node_max_probs = [max(float(x) for x in parts[3:]) for parts in node_data[node]]
        qualities.append(sum(node_max_probs) / len(node_max_probs) if node_max_probs else 0)
    return qualities

def get_log10_prob_none(nodes, node_data, target_aa):
    col_idx = get_aa_col_index(target_aa)
    log_probs = []
    for node in nodes:
        log_prob_none = 0.0
        has_certain_target = False
        for parts in node_data[node]:
            prob_target = float(parts[col_idx])
            if prob_target >= 1.0:
                has_certain_target = True
                break
            prob_no_target = 1.0 - prob_target
            if prob_no_target > 0:
                log_prob_none += math.log10(prob_no_target)
        log_probs.append(-100 if has_certain_target else log_prob_none)
    return log_probs

def sequence_resurection(nodes, node_data, target_aa):
    os.makedirs(output_folder, exist_ok=True)
    for node in nodes:
        seq_data = []
        for parts in node_data[node]:
            if len(parts) > 2:
                site = int(parts[1])
                original_state = parts[2]
                if original_state == target_aa:
                    current_probs = [(float(parts[3 + i]), aa) for i, aa in enumerate(AA_ORDER)]
                    current_probs.sort(key=lambda x: x[0], reverse=True)
                    seq_data.append((site, original_state, current_probs[1][1]))
                else:
                    seq_data.append((site, original_state, original_state))
        if not seq_data:
            continue
        seq_data.sort(key=lambda x: x[0])
        original_seq = "".join([x[1] for x in seq_data])
        modified_seq = "".join([x[2] for x in seq_data])
        
        orig_file = os.path.join(output_folder, f"{node}.fasta")
        if not os.path.exists(orig_file):
            with open(orig_file, "w") as f:
                f.write(f">{node}\n{original_seq}\n")
        if original_seq != modified_seq:
            mod_file = os.path.join(output_folder, f"{node}_replaced_{target_aa}.fasta")
            with open(mod_file, "w") as f:
                f.write(f">{node}_no_{target_aa}\n{modified_seq}\n")

# ==========================================
# 3. DATA COLLECTION
# ==========================================
data_exp = {aa: get_expectation_values(nodes_of_interest, node_data, aa) for aa in AA_ORDER}
data_avg = {aa: get_average_probabilities(nodes_of_interest, node_data, aa) for aa in AA_ORDER}
data_counts = {aa: get_certainty_counts(nodes_of_interest, node_data, aa) for aa in AA_ORDER}
data_log = {aa: get_log10_prob_none(nodes_of_interest, node_data, aa) for aa in AA_ORDER}
data_asr = get_asr_quality(nodes_of_interest, node_data)

data_raw_prob = {
    aa: [1.0 - (10**val if val != -100 else 0.0) for val in data_log[aa]] 
    for aa in AA_ORDER
}

group1_exp_sum = [sum(data_exp[aa][i] for aa in AA_GROUP_1) for i in range(len(nodes_of_interest))]
group2_exp_sum = [sum(data_exp[aa][i] for aa in AA_GROUP_2) for i in range(len(nodes_of_interest))]

for aa in AA_ORDER:
    sequence_resurection(nodes_of_interest, node_data, aa)

# ==========================================
# 4. PLOTTING HELPER FUNCTIONS
# ==========================================
x = range(len(nodes_of_interest))

def plot_group_subset(filename, title, ylabel, aa_list, data_dict, is_probability=False):
    """Vykreslí graf pro vybranou skupinu (Early nebo Late) s 10 UNIKÁTNÍMI BARVAMI v rámci daného grafu"""
    plt.figure(figsize=(9, 6))
    
    # Paleta přesně 10 unikátních barev pro 10 aminokyselin v daném grafu
    local_palette = cm.get_cmap('tab10', len(aa_list))
    
    for idx, aa in enumerate(aa_list):
        # Pokud jde o Tryptofan (W), vynutíme fialovou, jinak dáme unikátní barvu z palety
        color = 'purple' if aa == "W" else local_palette(idx)
        marker = markers_list[idx % len(markers_list)]
        
        plt.plot(x, data_dict[aa], marker=marker, color=color, linestyle='-', label=aa, linewidth=1.8)
    
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Ancestral Nodes", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.xticks(x, nodes_of_interest, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.6)
    if is_probability:
        plt.ylim([-0.05, 1.05])
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=300)
    plt.close()

def plot_solo_target(filename, title, ylabel, data_vector, color='purple', is_probability=False):
    """Vykreslí čistej sólo graf pouze pro Tryptofan (W)"""
    plt.figure(figsize=(9, 6))
    plt.plot(x, data_vector, marker='o', linewidth=2.5, color=color, label=f"{HIGHLIGHT_AA} (Trp)")
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Ancestral Nodes", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.xticks(x, nodes_of_interest, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.6)
    if is_probability:
        plt.ylim([-0.05, 1.05])
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=300)
    plt.close()

# ==========================================
# GENERATION: 1. EARLY AAs PLOTS
# ==========================================
plot_group_subset("1a_expectation_early.png", "Expectation Values Across Ancestral Nodes (Early AAs)", "Expected Residue Count", AA_GROUP_1, data_exp)
plot_group_subset("2a_counts_early.png", "Top-Ranked Residue Counts (Early AAs)", "Number of Positions", AA_GROUP_1, data_counts)
plot_group_subset("3a_log10_absence_early.png", "Log10 Probability of Absence (Early AAs)", "Log10 Probability", AA_GROUP_1, data_log)
plot_group_subset("4a_presence_prob_early.png", "Probability of Presence (Early AAs)", "Probability (0 to 1)", AA_GROUP_1, data_raw_prob, is_probability=True)

# ==========================================
# GENERATION: 2. LATE AAs PLOTS
# ==========================================
plot_group_subset("1b_expectation_late.png", "Expectation Values Across Ancestral Nodes (Late AAs)", "Expected Residue Count", AA_GROUP_2, data_exp)
plot_group_subset("2b_counts_late.png", "Top-Ranked Residue Counts (Late AAs)", "Number of Positions", AA_GROUP_2, data_counts)
plot_group_subset("3b_log10_absence_late.png", "Log10 Probability of Absence (Late AAs)", "Log10 Probability", AA_GROUP_2, data_log)
plot_group_subset("4b_presence_prob_late.png", "Probability of Presence (Late AAs)", "Probability (0 to 1)", AA_GROUP_2, data_raw_prob, is_probability=True)

# ==========================================
# GENERATION: 3. TRP (W) SOLO PLOTS
# ==========================================
plot_solo_target("TRP_1_expectation_value.png", "Expectation Value for Tryptophan (W)", "Expected Residue Count", data_exp[HIGHLIGHT_AA])
plot_solo_target("TRP_2_top_ranked_counts.png", "Top-Ranked Position Counts for Tryptophan (W)", "Number of Positions", data_counts[HIGHLIGHT_AA])
plot_solo_target("TRP_3_log10_absence.png", "Log10 Probability of Absence for Tryptophan (W)", "Log10 Probability", data_log[HIGHLIGHT_AA])
plot_solo_target("TRP_4_presence_probability.png", "Probability of Presence for Tryptophan (W)", "Probability (0 to 1)", data_raw_prob[HIGHLIGHT_AA], is_probability=True)

# ==========================================
# GENERATION: 4. GLOBAL & BASELINE PLOTS
# ==========================================
# ASR Quality
plt.figure(figsize=(9, 6))
plt.plot(x, data_asr, marker='D', color='black', linewidth=2, label="ASR Quality")
plt.title("Ancestral Sequence Reconstruction Quality", fontsize=14, pad=12)
plt.xlabel("Ancestral Nodes", fontsize=11)
plt.ylabel("Mean Max Posterior Probability", fontsize=11)
plt.xticks(x, nodes_of_interest, rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "GLOBAL_asr_quality.png"), dpi=300)
plt.close()

# TRP Mean Prob vs Baseline
plt.figure(figsize=(9, 6))
plt.plot(x, data_avg[HIGHLIGHT_AA], marker='o', linewidth=2.5, color='purple', label=f"{HIGHLIGHT_AA} (Trp)", zorder=5)
plt.axhline(y=PI_TRP, color='red', linestyle='--', linewidth=1.5, label=rf'Model Baseline $\pi_W$ ({PI_TRP*100:.1f}%)')
plt.title(f"Mean Posterior Probability of {HIGHLIGHT_AA} per Site (vs. $\pi_W$ Baseline)", fontsize=14, pad=12)
plt.xlabel("Ancestral Nodes", fontsize=11)
plt.ylabel("Mean Posterior Probability", fontsize=11)
plt.xticks(x, nodes_of_interest, rotation=45, ha='right')
plt.minorticks_on()
plt.grid(True, which='major', linestyle='--', alpha=0.7)
plt.grid(True, which='minor', linestyle=':', alpha=0.45)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "GLOBAL_trp_mean_prob_vs_baseline.png"), dpi=300)
plt.close()

# Group Comparison
fig7, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
ax_top.plot(x, group1_exp_sum, marker='o', linewidth=2.5, color='teal', label=GROUP_1_NAME)
ax_top.set_title(f"Amino Acid Group Comparison: {GROUP_1_NAME} vs. {GROUP_2_NAME}", fontsize=14, pad=12)
ax_top.set_ylabel(f"Expected Count ({GROUP_1_NAME})", fontsize=11)
ax_top.grid(True, linestyle='--', alpha=0.6)
ax_top.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

ax_bot.plot(x, group2_exp_sum, marker='s', linewidth=2.5, color='darkorange', label=GROUP_2_NAME)
ax_bot.set_xlabel("Ancestral Nodes", fontsize=11)
ax_bot.set_ylabel(f"Expected Count ({GROUP_2_NAME})", fontsize=11)
ax_bot.set_xticks(x)
ax_bot.set_xticklabels(nodes_of_interest, rotation=45, ha='right')
ax_bot.grid(True, linestyle='--', alpha=0.6)
ax_bot.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "GLOBAL_group_comparison.png"), dpi=300)
plt.close()

print(f"\nVšechny grafy byly úspěšně vygenerovány s unikátními barvami a uloženy do:\n{output_folder}")
