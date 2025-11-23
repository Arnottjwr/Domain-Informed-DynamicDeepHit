# Domain-Informed Dynamic DeepHit for Chronic Kidney Disease

This repository contains the implementation of Domain-Informed Dynamic DeepHit for predicting outcomes associated with Chronic Kidney Disease (CKD).


## Overview

D3-CKD is a machine learning model trained on time-series electronic health records to predict CKD-related outcomes. The model integrates medical domain knowledge to improve survival predictions. For full details, see here .

## Background

The code is based on the uninformed model from [this repository](https://github.com/georgehc/survival-intro), with significant modifications to incorporate domain knowledge and dynamic predictions.

- Python 3.11

### Setup

1. **Create a virtual environment:**
```bash
python3 -m venv venv
```

2. **Activate the virtual environment:**

On Linux/macOS:
```bash
source venv/bin/activate
```

On Windows (Command Prompt):
```bash
venv\Scripts\activate
```

On Windows (PowerShell):
```bash
venv\Scripts\Activate.ps1
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```


## Configuration

Model settings and experimental parameters are configured in `config.yaml` in the root directory.

### Model Architecture Parameters

- **`layers_for_attention`**: Number and width of layers in the attention component's FCNN
- **`layers_for_each_deephit_event`**: Number and width of layers in each FCNN subnetwork for the final component
- **`layers_for_predicting_next_time_step`**: Number and width of layers in the FCNN that predicts the next time step of dynamic covariates from RNN output
- **`num_hidden`**: Number of features in the RNN hidden state
- **`num_rnn_layers`**: Number of RNN layers
- **`rnn_type`**: RNN architecture (supported: LSTM, GRU, vanilla RNN)
- **`dropout`**: Dropout probability


## To Do

- Add functionality to plot domain loss functions (separately and combined)
- Fix C-index score calculation (potentially caused by synthetic data)
- Implement automatic population of events and features arrays in config from data file (pass data path instead of manual specification)
