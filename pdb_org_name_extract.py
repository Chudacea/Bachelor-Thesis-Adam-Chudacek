import os
import re

input_folder = ""
output_folder = ""

def sanitize_name(name: str) -> str:
    """
    Replace problematic characters with underscores so the string is safe for filenames.
    """
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)

def extract_org_name(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".pdb"):
            continue

        input_path = os.path.join(input_folder, filename)
        with open(input_path, "r") as f:
            all_lines = f.readlines()

        org_name = None
        molecule_name = None

        for line in all_lines:
            if line.startswith("SOURCE") and "ORGANISM_SCIENTIFIC:" in line:
                parts = line.split("ORGANISM_SCIENTIFIC:")[-1].strip()
                if ";" in parts:
                    parts = parts.split(";")[0]
                org_name = parts.strip()

            elif line.startswith("COMPND") and "MOLECULE:" in line:
                parts = line.split("MOLECULE:")[-1].strip()
                if ";" in parts:
                    parts = parts.split(";")[0]
                molecule_name = parts.strip()

        if not org_name:
            org_name = "UnknownOrg"
        if not molecule_name:
            molecule_name = "UnknownMol"

        # sanitize names
        org_name = sanitize_name(org_name)
        molecule_name = sanitize_name(molecule_name)

        base = filename
        if base.endswith("-F1-model_v4.pdb"):
            base = base.replace("-F1-model_v4.pdb", "")

        new_filename = f"{molecule_name}_{base}_{org_name}.pdb"
        new_filename = sanitize_name(new_filename)  # final safety check

        output_path = os.path.join(output_folder, new_filename)

        with open(output_path, "w") as out_f:
            out_f.writelines(all_lines)

        print(f"Processed {filename} -> {new_filename}")

extract_org_name(input_folder, output_folder)
