import pubchempy as pcp
import pandas as pd


mix_data = pd.read_csv("datasets/competition/Mixure_Definitions_Training_set.csv")

cid_columns = [col for col in mix_data.columns if col.startswith("CID")]

all_cids = []
for col in cid_columns:
    all_cids.extend(mix_data[col].unique())

all_cids = set(all_cids)

cid_to_smi = {}

for cid in all_cids:
    if cid == 0:
        cid_to_smi[0] = None
    else:
        cid_to_smi[cid] = pcp.Compound.from_cid(int(cid)).isomeric_smiles

smi_data = pd.DataFrame()
smi_data["Dataset"] = mix_data["Dataset"]
smi_data["Mixture Label"] = mix_data["Mixture Label"]
for i, col in enumerate(cid_columns):
    smi_data[f"smi_{i}"] = mix_data[col].map(cid_to_smi)

smi_data.to_csv("datasets/competition/Mixture_definitions_smi_training.csv", index=False)