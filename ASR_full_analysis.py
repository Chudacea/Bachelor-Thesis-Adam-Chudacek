import math
import os
import matplotlib.pyplot as plt

# --- ZAJÍMAVÝ NODES ---
# pro W8TRN9, nodes_of_interest = ["Anc_99", "Anc_98", "Anc_97", "Anc_96", "Anc_85", "Anc_68", "Anc_41", "Anc_39", "Anc_21"]



# --- KONFIGURACE ---
input_file = "/Users/adamchudacek/Documents/Marburg/ASR/fbaA/ezASR_results_copy/alligned_hits_trimmed.state"  # Změň na svůj soubor
output_folder = "/Users/adamchudacek/Documents/Marburg/ASR/fbaA/RESURECTION"
nodes_of_interest = ["Anc_99", "Anc_98", "Anc_97", "Anc_96", "Anc_85", "Anc_68", "Anc_41", "Anc_39", "Anc_21"]  # Doplň všechny nody pro x-osu
aas_of_interest = ["F", "Y", "W", "H", "C", "M", "N", "Q", "K", "R"]  # Seznam všech AA pro porovnání

AA_ORDER = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

def get_aa_col_index(aa):
    if aa not in AA_ORDER:
        raise ValueError(f"Unknown Amino Acid: {aa}")
    return 3 + AA_ORDER.index(aa)

# ==========================================
# 1. OPTIMALIZACE: Načtení dat jen jednou
# ==========================================
print("Načítám a parsuji soubor (tohle potrvá jen chvilku)...")
node_data = {node: [] for node in nodes_of_interest}

with open(input_file, "r") as f:
    lines = f.readlines()[9:]
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        node = parts[0]
        # Uložíme si už rozdělená data jen pro ty uzly, které nás zajímají
        if node in node_data:
            node_data[node].append(parts)

print("Data načtena. Počítám parametry pro grafy...")

# ==========================================
# 2. VÝPOČETNÍ FUNKCE (Nyní extrémně rychlé)
# ==========================================
def get_expectation_values(nodes, node_data, target_aa):
    col_idx = get_aa_col_index(target_aa)
    exp_values = []
    for node in nodes:
        # Sečte pravděpodobnosti dané AA ze sloupce
        exp_val = sum(float(parts[col_idx]) for parts in node_data[node])
        exp_values.append(exp_val)
    return exp_values

def get_certainty_counts(nodes, node_data, target_aa):
    counts = []
    for node in nodes:
        count = sum(1 for parts in node_data[node] if parts[2] == target_aa)
        counts.append(count)
    return counts

def get_asr_quality(nodes, node_data):
    qualities = []
    for node in nodes:
        node_max_probs = [max(float(x) for x in parts[3:]) for parts in node_data[node]]
        if node_max_probs:
            qualities.append(sum(node_max_probs) / len(node_max_probs))
        else:
            qualities.append(0)
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
        
        if has_certain_target:
            log_probs.append(-100) # Umělá hodnota znamenající 100% jistotu přítomnosti
        else:
            log_probs.append(log_prob_none)
    return log_probs

def sequence_resurection(nodes, node_data, target_aa):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for node in nodes:
        seq_data = []
        for parts in node_data[node]:
            if len(parts) > 2:
                site = int(parts[1])
                original_state = parts[2]
                
                if original_state == target_aa:
                    current_probs = [(float(parts[3 + i]), aa) for i, aa in enumerate(AA_ORDER)]
                    current_probs.sort(key=lambda x: x[0], reverse=True)
                    second_best_aa = current_probs[1][1]
                    seq_data.append((site, original_state, second_best_aa))
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
# 3. SBĚR DAT A VYKRESLENÍ (Beze změny)
# ==========================================
data_exp = {aa: get_expectation_values(nodes_of_interest, node_data, aa) for aa in aas_of_interest}
data_counts = {aa: get_certainty_counts(nodes_of_interest, node_data, aa) for aa in aas_of_interest}
data_log = {aa: get_log10_prob_none(nodes_of_interest, node_data, aa) for aa in aas_of_interest}
data_asr = get_asr_quality(nodes_of_interest, node_data)

data_raw_prob = {}
for aa in aas_of_interest:
    data_raw_prob[aa] = [1.0 - (10**val if val != -100 else 0.0) for val in data_log[aa]]

for aa in aas_of_interest:
    sequence_resurection(nodes_of_interest, node_data, aa)

# --- PLOTTING ---
fig, axs = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle("Vývoj ASR parametrů přes uzly (Nodes)", fontsize=16)

x = range(len(nodes_of_interest))

for aa in aas_of_interest:
    axs[0, 0].plot(x, data_exp[aa], marker='o', label=aa)
axs[0, 0].set_title("Expectation Value")
axs[0, 0].set_ylabel("Očekávaný počet AA")
axs[0, 0].legend()

for aa in aas_of_interest:
    axs[0, 1].plot(x, data_counts[aa], marker='s', label=aa)
axs[0, 1].set_title("Počet pozic jako vítězná AA")
axs[0, 1].set_ylabel("Počet pozic")
axs[0, 1].legend()

for aa in aas_of_interest:
    axs[1, 0].plot(x, data_log[aa], marker='^', label=aa)
axs[1, 0].set_title("Log10 pravděpodobnost absence AA")
axs[1, 0].set_ylabel("Log10 Prob (nižší = jistější přítomnost)")
axs[1, 0].legend()

for aa in aas_of_interest:
    axs[1, 1].plot(x, data_raw_prob[aa], marker='v', label=aa)
axs[1, 1].set_title("Pravděpodobnost přítomnosti (alespoň 1×)")
axs[1, 1].set_ylabel("Pravděpodobnost (0 až 1)")
axs[1, 1].set_ylim([-0.05, 1.05])
axs[1, 1].legend()

axs[2, 0].plot(x, data_asr, marker='D', color='black', label="ASR Quality")
axs[2, 0].set_title("Kvalita rekonstrukce (ASR Quality)")
axs[2, 0].set_ylabel("Průměrná Max pravděpodobnost")
axs[2, 0].legend()

axs[2, 1].axis('off')

for ax in axs.flat:
    if ax.has_data():
        ax.set_xticks(x)
        ax.set_xticklabels(nodes_of_interest, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()