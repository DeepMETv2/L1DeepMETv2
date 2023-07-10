# L1DeepMETv2
**GNN** based algorithm for **online** MET reconstruction for CMS L1 trigger. Regress MET from _L1Puppi candidates_.

This work extends the following efforts for MET reconstruction algorithms:
- **Fully-Connected Neural Network (FCNN)** based algorithm in `keras` for **offline** MET reconstruction: [DeepMETv1](https://github.com/DeepMETv2/DeepMETv1)
- **Graph Neural Network (GNN)** based algorithm in `torch` for **offline** MET reconstruction: [DeepMETv2](https://github.com/DeepMETv2/DeepMETv2)
- **FCNN** based algorithm in `keras` for **online** MET reconstruction: [L1MET](https://github.com/jmduarte/L1METML) 



## Prerequisites 

```
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
```


## Produce Input Data

For producing training input data, we use _TTbar process_ simulation data available in [this link](https://cernbox.cern.ch/files/link/public/JK2InUjatHFxFbf?tiles-size=1&items-per-page=100&view-mode=resource-table). The data files are in `.root` format and they contain _L1_ information; the full list of variables available in the `.root` files can be found in [`./data_ttbar/branch_list_L1_TTbar.csv`](https://github.com/DeepMETv2/L1DeepMETv2/blob/master/data_ttbar/branch_list_L1_TTbar.csv). From the list of variables, we extract the ones that will be used for our training and save it to `.npz` format under `./data_ttbar/raw/`, using [`./data_ttbar/generate_npz.py`](https://github.com/DeepMETv2/L1DeepMETv2/blob/master/data_ttbar/generate_npz.py). 

```
L1PuppiCands_pt, L1PuppiCands_eta, L1PuppiCands_phi, L1PuppiCands_puppiWeight, L1PuppiCands_pdgId, L1PuppiCands_charge 
```

From these six variables, we make the following training inputs into `.pt` files in `./data_ttbar/processed/` using [`./model/data_loader.py`](https://github.com/DeepMETv2/L1DeepMETv2/blob/master/model/data_loader.py):

```
pt, px (= pt*cos(phi)), py (= pt*sin(phi)), eta, phi, puppiWeight, pdgId, charge 

```


## Get Input Data and Train 

Training script [`train.py`](https://github.com/DeepMETv2/L1DeepMETv2/blob/master/train.py) will use input data using torch dataloader. Training input in `.npz` format is available in [this link](https://cernbox.cern.ch/s/RETpE7fzw4g0lnF) under `/raw/`.

1. Download the `/raw/` folder containing `.npz` files and place the folder under `./data_ttbar/`.
2. When you run the training script right after step 1, the training script will fetch the `.npz` files from `./data_ttbar/raw` and produce dataloader from these files, using the functions defined in [`./model/data_loader.py`](https://github.com/DeepMETv2/L1DeepMETv2/blob/master/model/data_loader.py). This dataloader-producing takes some time, so feel free to use the dataloader already produced and saved in [this link](https://cernbox.cern.ch/s/RETpE7fzw4g0lnF). From the link, download the folder `processed` with dataloader saved as `.pt` files and place them under `./data_ttbar/`. 
3. Run the training script `train.py` using the following command:
```
python train.py --data data_ttbar --ckpts ckpts_ttbar
```
If you have done the step 2, the training script will directly fetch the input dataloader from `./data_ttbar/processed/` and save the training & evaluation output to `./ckpts_ttbar`.

