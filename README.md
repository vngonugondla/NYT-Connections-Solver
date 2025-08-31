# LLM-based NYT Connections Solver

This repository contains code for a NYT Connections solver that leverages Large Language Models (LLMs) with different prompting strategies and decoding methods.

## Overview

This repository explores the effectiveness of different LLM approaches for solving NYT Connections game. It compares zero-shot, one-shot, and few-shot prompting strategies combined with different decoding methods (greedy and beam search) to determine the most effective combination.

## Repository Structure

### Core Files

- **baseline.py**: Implements and tests GPT-4 in both zero-shot and few-shot settings for NYT Connections solving.

- **connectLLMSolver.py**: Generates one-shot and few-shot candidate solutions for NYT Connections puzzles.

- **Seq_Seq_Finetuning.ipynb**: Jupyter notebook that finetunes a FLAN-T5 model on NYT Connections puzzles and compares its performance against baseline approaches.

- **test.py**: Runs comprehensive tests across all combinations of candidate generation methods and solver strategies:
  - All Candidates + Greedy
  - All Candidates + Beam Search
  - Zero-shot Candidate Generation + Greedy
  - Zero-shot Candidate Generation + Beam Search
  - Few-shot Candidate Generation + Greedy
  - Few-shot Candidate Generation + Beam Search

### Supporting Files

- **data.py**: Contains data processing utilities for NYT Connections puzzles.

- **few-shot_prompting.py**: Implements few-shot prompting strategies for NYT Connections.

- **nyt_dataset.json**: Dataset containing NYT Connections puzzles for training and evaluation.

- **requirements.txt**: Lists Python package dependencies for the project.

- **environment.yml**: Conda environment specification for reproducing the project environment.

## Getting Started

1. Set up the environment:
   ```
   conda env create -f environment.yml
   ```
   or
   ```
   pip install -r requirements.txt
   ```

2. Test GPT-4 effectiveness:
   ```
   python baseline.py
   ```
   This will evaluate GPT-4's performance in both zero-shot and few-shot settings on NYT Connections puzzles.

3. Run comprehensive tests:
   ```
   python test.py
   ```
   This will test all combinations of candidate generation methods and decoding strategies.

4. Explore the Seq_Seq_Finetuning.ipynb notebook to understand the finetuning process and results.

## Results

The project compares different approaches to solving NYT Connections using LLMs, with detailed results available in the test outputs and the Seq_Seq_Finetuning.ipynb notebook.
