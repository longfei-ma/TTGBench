# Project Name

This repository contains the implementation of various methods for enhancing and predicting tasks using Large Language Models (LLMs) on dynamic graph datasets. The codebase is organized to support both LLMs-as-Enhancers and LLMs-as-Predictors approaches. This README provides instructions on how to set up and use the code.

## Prerequisites

- Python 3.9+
- CUDA-enabled GPU (for training and testing)
- Required Python packages (install via `pip install -r requirements.txt`)
- Bash shell for running scripts

## Dataset Setup

1. Download the required dataset from according to specifications(https://ttgbench.netlify.app ).
2. Place the downloaded dataset files in the `datasets/` directory.

## Usage Instructions

The codebase is divided into two main categories: **LLMs-as-Enhancers** and **LLMs-as-Predictors**. Follow the instructions below to run the experiments.

### 1. LLMs-as-Enhancers

Navigate to the `DyGLLM` directory:

```bash
cd DyGLLM
```

#### 1.1 Link Prediction Training

Run the link prediction training script with the following command:

```bash
bash scripts/train_link_ablation.sh FOOD TGN sbert 0 72 2
```

#### 1.2 Link Prediction Testing

Run the link prediction testing script:

```bash
bash scripts/test_link_abla.sh FOOD 0 72 sbert
```

#### 1.3 Node Classification Training

Run the node classification training script:

```bash
bash scripts/train_node_simple.sh Beeradvocate JODIE sbert 0 72
```

#### 1.4 Node Classification Testing

Run the node classification testing script:

```bash
bash scripts/test_node_dataset.sh IMDB 0 sbert 72
```

### 2. LLMs-as-Predictors

This category includes three methods: **LLaGA**, **GraphGPT**, and **TempPrompt**.

#### 2.1 LLaGA

Navigate to the `LLaGA` directory:

```bash
cd LLaGA
```

##### LLaGA-ND (Node-level Dynamic)

- **Training**:

  ```bash
  CUDA_VISIBLE_DEVICES=2 nohup bash scripts/train.sh vicuna $task $dataset 16 sbert 72 0 sbert
  ```

  - `$task`: Either `lp` (link prediction) or `nc` (node classification).
  - `$dataset`: Specify the dataset to evaluate (e.g., `FOOD`).

- **Testing**:

  ```bash
  CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test1_run.sh vicuna lp FOOD 32 sbert 72 0 sbert &
  ```

##### LLaGA-HO (High-Order)

- **Training**:

  ```bash
  CUDA_VISIBLE_DEVICES=2 nohup bash scripts/train.sh vicuna_4hop lp FOOD 16 sbert 72 0 sbert &
  ```

- **Testing**:

  ```bash
  CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test1_run.sh vicuna_4hop lp FOOD 32 sbert 72 0 sbert &
  ```

#### 2.2 GraphGPT

Navigate to the `GraphGPT` directory:

```bash
cd GraphGPT
```

- **Training**:

  ```bash
  CUDA_VISIBLE_DEVICES=3 nohup bash scripts/tune_script/graphgpt_stage2.sh FOOD lp sbert 72 0 sbert &
  ```

- **Testing**:

  ```bash
  CUDA_VISIBLE_DEVICES=3 nohup bash scripts/eval_script/graphgpt_eval.sh FOOD lp sbert 72 0 sbert &
  ```

#### 2.3 TempPrompt

In the `GraphGPT` directory, run the TempPrompt evaluation:

```bash
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/eval_script/temprompt1.sh FOOD lp 72 0 nondst2 sbert &
```

## Notes

- Ensure that the dataset name and task parameters are correctly set for your specific use case.
- The scripts assume access to a CUDA-enabled GPU. Modify `CUDA_VISIBLE_DEVICES` as needed based on your hardware configuration.
- For background execution, the `nohup` command is used to prevent process termination upon terminal closure.
- Check the `scripts/` directory for additional configuration options or script details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bug reports, feature requests, or other suggestions.

## License

This project is licensed under the [MIT License](LICENSE).