# Hitting Stride by Degrees: Fine Grained Molecular Generation via Diffusion Model

In this work, we propose a diffusion-based molecular generation model called DIFFUMOL. Unlike the traditional diffusion model, DIFFUMOL
selectively adds Gaussian noise to retain conditional features as guidance information for the generation of user-specified  molecules.  DIFFUMOL can effectively fit the data distribution of the dataset and perform unconditional molecular generation tasks. DIFFUMOL can conduct conditional molecular generation tasks based on properties and scaffolds. Furthermore, DIFFUMOL is capable of property optimization tasks based on existing molecules.

##  Datasets And Weights

The datasets and trained model weights used in the paper can be obtained from the following link. Place the downloaded datasets in the `./datasets` directory and the weight files in the `./weights` directory to ensure the normal training of the model. 

https://drive.google.com/drive/folders/13jI6sYG8D7XH2ksjiks1rymMfWl8K2U1?usp=sharing

##  Environment Setup

An environment for DIFFUMOL can be easily setup via Anaconda:

```shell
git clone https://github.com/wengong-jin/multiobj-rationale.git
cd DIFFUMOL
conda create -n DIFFUMOL python=3.8
conda activate DIFFUMOL
pip install -r requirements.txt
```

## Train

All tasks in DIFFUMOL are controlled by the `./diffumol/config.json` configuration file, making our code easy to run and train. After modifying the configuration file, run

```shell
nohup python -u train.py >Train.log 2>&1 &
```

## Generation

The trained weights and related training configurations will be saved. Execute the following code to generate a CSV file for the target task, including the relevant molecular property information.

```shell
python generate.py --model_path <trained_model_path> --batch_size <batch_size>  --sample <number of molecules generated> --out_dir <save_csv_path>
```

##  Evaluation

More detailed evaluation results can be obtained by executing the following code.

```shell
python evaluate/moses_test.py --path <path of the csv file>
```

