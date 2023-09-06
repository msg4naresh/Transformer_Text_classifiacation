# Fake News Classification using Transformer-based Models

This project aims to build a deep learning model to classify fake news articles. 

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
   - [Create a Virtual Environment](#create-a-virtual-environment)
   - [Activate the Virtual Environment](#activate-the-virtual-environment)
   - [Install Libraries](#install-libraries)
3. [Training and Evaluating the Model](#training-and-evaluating-the-model)

---

## Requirements

- Python 3.x
- pip
- A virtual environment (optional but recommended)

---

## Installation

### Create a Virtual Environment

It's recommended to create a virtual environment to isolate dependencies. Use the following command to create a new virtual environment inside a directory named `.venv`:

```bash
python -m venv .venv
```

### Activate the Virtual Environment

After creating the virtual environment, it needs to be activated:

- **For macOS and Linux:**

    ```bash
    source .venv/bin/activate
    ```

- **For Windows:**

    ```bash
    .\.venv\Scripts\activate
    ```

### Install Libraries

After activating the virtual environment, install the required libraries:

```bash
pip install -r requirements.txt
```

## Training and Evaluating the Model

Once the libraries are installed, you can execute the following command to train and evaluate your deep learning model:

```bash
python distillBertTransformer_FakeNewsClassification.py
```

> **Note:** The sample size is currently limited to 1000 due to computational constraints.
