# Generalized AI-based Model for Non-Residential Buildings

This project implements a generalized machine learning model using LSTM (Long Short-Term Memory) neural networks to predict energy consumption patterns in non-residential buildings. The solution uses the Building Data Genome Project dataset to train and evaluate the model.

## Dataset

### Download Instructions

1. Visit the Kaggle dataset page: [Building Data Genome Project v1](https://www.kaggle.com/datasets/claytonmiller/building-data-genome-project-v1) (Last accessed: 23.12.2025)

2. Download the dataset using one of these methods:

    - **Kaggle Website**: Go to the link above and click "Download"
    - **Kaggle CLI**:
        ```bash
        kaggle datasets download -d claytonmiller/building-data-genome-project-v1
        ```

3. Extract the downloaded files and place the raw data in the `data/raw/` folder:
    ```
    data/
    └── raw/
        ├── meta_open.csv
        └── [other dataset files]
    ```

## Project Structure

```
Generalized-AI-based-model-for-non-residential-buildings/
├── data/
│   └── raw/                          # Raw dataset files from Kaggle
├── ploting/
│   └── utlis/                        # Plotting utilities
├── 1_process_data.ipynb              # Data processing and cleaning
├── 2_prepare_data.ipynb              # Data preparation and normalization
├── 3_find_lstm_size.ipynb            # LSTM architecture optimization
├── 4_evaluate_lstm_size.ipynb        # Evaluation of different LSTM sizes
├── 5_optimize_hyperparameters.ipynb  # Hyperparameter optimization
├── 6_evaluate_hyperparameters_1.ipynb # Hyperparameter evaluation (part 1)
├── 6_evaluate_hyperparameters_2.ipynb # Hyperparameter evaluation (part 2)
├── 7_evaluate_best_model.ipynb       # Final model evaluation
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Installation

### Prerequisites

-   Python 3.12 or later
-   pip (Python package manager)

### Setup Instructions

1. **Clone/Download the repository**:

    ```bash
    cd path/to/Generalized-AI-based-model-for-non-residential-buildings
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download and prepare the dataset** (see Dataset section above)

## Running the Solution

The notebooks should be executed in the following order:

### 1. **Data Processing** (`1_process_data.ipynb`)

-   Reads raw data from `data/raw/meta_open.csv`
-   Cleans and validates the data
-   Handles missing values and data inconsistencies
-   Outputs processed data

### 2. **Data Preparation** (`2_prepare_data.ipynb`)

-   Normalizes data types (float32 optimization)
-   Creates train/validation/test splits
-   Performs feature engineering
-   Prepares data for LSTM training

### 3. **Find LSTM Size** (`3_find_lstm_size.ipynb`)

-   Tests different LSTM unit configurations
-   Evaluates performance metrics (MSE, MAE, MAPE)
-   Identifies optimal LSTM architecture
-   Runs for 100 epochs with batch size 1024

### 4. **Evaluate LSTM Size** (`4_evaluate_lstm_size.ipynb`)

-   Detailed evaluation of various LSTM sizes
-   Analyzes training/validation/test performance
-   Generates comparison plots and visualizations
-   Identifies the best performing architecture

### 5. **Optimize Hyperparameters** (`5_optimize_hyperparameters.ipynb`)

-   Tests different activation functions (tanh, sigmoid)
-   Optimizes dropout rates (0.1, 0.2)
-   Tests recurrent dropout configurations
-   Trains multiple model variants

### 6. **Evaluate Hyperparameters** (`6_evaluate_hyperparameters_1.ipynb` & `6_evaluate_hyperparameters_2.ipynb`)

-   Analyzes results from hyperparameter optimization
-   Compares performance metrics across configurations
-   Generates evaluation plots and statistics
-   Identifies best hyperparameter combinations

### 7. **Evaluate Best Model** (`7_evaluate_best_model.ipynb`)

-   Loads the best performing model
-   Final evaluation on test set
-   Generates comprehensive performance report
-   Produces visualization of predictions vs. actual values

## Dependencies

See `requirements.txt` for the complete list of dependencies:

-   **pandas**: Data manipulation and analysis
-   **numpy**: Numerical computing
-   **matplotlib**: Data visualization
-   **tensorflow**: Deep learning framework
-   **scikit-learn**: Machine learning utilities
-   **jupyter**: Interactive notebook environment

## Output Data

The notebooks generate the following output files:

-   Processed and prepared datasets (CSV format)
-   Trained LSTM models (Keras format)
-   Training history and metrics (JSON format)
-   Evaluation plots and visualizations (PNG format)

## System Requirements

-   **RAM**: Minimum 32GB (64GB+ recommended)
-   **GPU**: Optional but recommended for faster training (NVIDIA CUDA-compatible GPU)
-   **Storage**: 50GB+ for dataset and trained models

## References

-   Building Data Genome Project v1: https://www.kaggle.com/datasets/claytonmiller/building-data-genome-project-v1
-   TensorFlow/Keras Documentation: https://www.tensorflow.org/
-   LSTM Networks: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
