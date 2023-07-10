# L1DeepMETv2
**GNN** based algorithm for **online** MET reconstruction at CMS L1 trigger level. Regress MET from _L1Puppi candidates_.

This work extends the following efforts for MET reconstruction algorithms:
- **Fully-connected Neural Network (FCNN)** based algorithm (`keras`) for **offline** MET reconstruction: [DeepMETv1](https://github.com/DeepMETv2/DeepMETv1)
- **Graph Neural Network** based algorithm (`torch`) for **offline** MET reconstruction: [DeepMETv2](https://github.com/DeepMETv2/L1DeepMETv2)
- **FCNN** based algorithm (`keras`) for **online** MET reconstruction: [L1MET](https://github.com/jmduarte/L1METML/tree/main) 



## Prerequisites 

<pre>
conda install cudatoolkit=10.2
conda install -c pytorch pytorch=1.9.0
export CUDA="cu102"
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-geometric
pip install coffea
pip install mplhep
</pre>



## Produce Input Data

For producing training input data, we use _TTbar process_ simulation data available in [this link](https://cernbox.cern.ch/files/link/public/JK2InUjatHFxFbf?tiles-size=1&items-per-page=100&view-mode=resource-table). The data are in `.root` format and they contain _L1-level_ information; the full list of information available in the `.root` files can be found in `./data_ttbar/branch_list_L1_TTbar.csv`. From the list of variables in the root files, we extract the ones that will be used for our training and save it to `.npz` format under `./data_ttbar/raw/` using `./data_ttbar/generate_npz.py`. 

```
L1PuppiCands_pt, L1PuppiCands_eta, L1PuppiCands_phi, L1PuppiCands_puppiWeight, L1PuppiCands_pdgId, L1PuppiCands_charge 
```

From these six variables, we make the following training inputs into `.pt` files in `./data_ttbar/processed/` using `./model/data_loader.py`:

```
pt, px (= pt*cos(phi)), py (= pt*sin(phi)), eta, phi, puppiWeight, pdgId, charge 

```


## Get Input Data and Train 

Training script `train.py` will use input data using torch dataloader. Training input in raw `.npz` format is available in [this link](https://cernbox.cern.ch/files/link/public/8anXECECjwH1iJy/TTbar_npz?items-per-page=100&view-mode=resource-table&tiles-size=1). 

1. Download the `.npz` files and place them under `./data_ttbar/raw/`.
2. `torch` will fetch the data and produce dataloader from these `.npz` files. This step 2 takes some time, so feel free to use the dataloader already processed and saved in [this link](). From the link, download the folder with dataloader saved as `.pt` files and place them under `./data_ttbar/processed/`. 
3. Run the training script `train.py` using the following command:
```
python train.py --data data_ttbar --ckpts ckpts_ttbar
```
It will fetch the input data from `./data_ttbar` and save the training/evaluation output to `./ckpts_ttbar`.


