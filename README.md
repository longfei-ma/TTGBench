# TTGBench

This is the source code for NeurIPS 2025 Datasets & Benchmarks submitted paper [TTGBench: Text-attributed Temporal Graph Benchmark for Temporal Graph Representation Learning with Large Language Models](https://openreview.net/forum?id=zae0fR82lS).

## Prerequisites

- Python 3.9+
- CUDA-enabled GPU (for training and testing)
- Required Python packages (install via `pip install -r requirements.txt`)
- Bash shell for running scripts

## Dataset Setup

1. Download the required dataset from according to specifications from our supported online [leaderboard](https://ttgbench.netlify.app).
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
bash scripts/train_link.sh FOOD TGN sbert 0 72
```

#### 1.2 Link Prediction Testing

Run the link prediction testing script:

```bash
bash scripts/test_link.sh FOOD 0 72 sbert
```

#### 1.3 Node Classification Training

Run the node classification training script:

```bash
bash scripts/train_node.sh Beeradvocate JODIE sbert 0 72
```

#### 1.4 Node Classification Testing

Run the node classification testing script:

```bash
bash scripts/test_node.sh IMDB 0 sbert 72
```

### 2. LLMs-as-Predictors

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
  CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test.sh vicuna lp FOOD 32 sbert 72 0 sbert &
  ```

##### LLaGA-HO (High-Order)

- **Training**:

  ```bash
  CUDA_VISIBLE_DEVICES=2 nohup bash scripts/train.sh vicuna_4hop lp FOOD 16 sbert 72 0 sbert &
  ```

- **Testing**:

  ```bash
  CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test.sh vicuna_4hop lp FOOD 32 sbert 72 0 sbert &
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

Navigate to the `TempPrompt` directory:

```bash
cd TempPrompt
```

Run the TempPrompt evaluation:

```bash
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/temprompt.sh FOOD lp 72 0 nondst2 sbert &
```

## Notes

- Ensure that the dataset name and task parameters are correctly set for your specific use case.
- The scripts assume access to a CUDA-enabled GPU. Modify `CUDA_VISIBLE_DEVICES` as needed based on your hardware configuration.
- Check the `scripts/` directory for additional configuration options or script details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bug reports, feature requests, or other suggestions.

## License

This project is licensed under the [MIT License](LICENSE).