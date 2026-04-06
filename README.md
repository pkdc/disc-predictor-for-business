# DISC Predictor for Business Communication

A machine learning system that predicts DISC personality styles from business email text using BERT embeddings and XGBoost classification. This tool helps organizations improve workplace communication by understanding the behavioral tendencies reflected in written correspondence.

## What is DISC?

DISC is a behavioral assessment framework that categorizes communication styles into four primary types:

| Style | Name | Characteristics |
|-------|------|-----------------|
| **D** | Dominance | Direct, results-oriented, decisive, competitive |
| **I** | Influence | Enthusiastic, collaborative, optimistic, persuasive |
| **S** | Steadiness | Patient, reliable, team-oriented, supportive |
| **C** | Conscientiousness | Analytical, detail-focused, systematic, accurate |

This model supports multi-label classification, recognizing that individuals often exhibit blends of multiple styles.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Data Pipeline](#data-pipeline)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [License](#license)

---

## Features

- **Robust Email Preprocessing**: Handles noisy corporate emails with 20+ regex patterns for headers, signatures, forwarded content, URLs, phone numbers, and more
- **BERT Embeddings**: Generates 768-dimensional semantic vectors using TensorFlow Hub's BERT encoder
- **Multi-Label Classification**: XGBoost wrapped in `MultiOutputClassifier` for predicting multiple DISC styles simultaneously
- **Hybrid Labeling Strategy**: Combines manual labels, rule-based keyword matching, and pseudo-labeling for scalable training data
- **Streamlit Web Interface**: Interactive MVP for real-time predictions
- **Docker Support**: Containerized deployment with docker-compose

---

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/disc-predictor-for-business.git
cd disc-predictor-for-business

# Build and run with Docker
docker-compose up --build

# Access the application at http://localhost:8501
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/disc-predictor-for-business.git
cd disc-predictor-for-business

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## Installation

### Prerequisites

- Python 3.12+
- pip or conda
- Docker (optional, for containerized deployment)

### Dependencies

Core dependencies include:

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | >=2.16.1 | Deep learning backend |
| tensorflow-hub | >=0.12.0 | Pre-trained BERT models |
| tensorflow-text | >=2.7.0 | Text preprocessing |
| xgboost | >=3.0.0 | Gradient boosting classifier |
| scikit-learn | >=1.2.2 | ML utilities and MultiOutputClassifier |
| streamlit | >=1.20.0 | Web interface |
| pandas | >=1.5.3 | Data manipulation |
| beautifulsoup4 | >=4.12.0 | HTML parsing for email cleaning |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Web Interface

Launch the Streamlit application:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser. Enter email text in the input field and click "Predict" to see the DISC classification.

### Programmatic Usage

```python
from utils.model_loader import load_model
from utils.preprocess import clean_msg_body
from utils.bert_embedder import get_bert_embeddings
from utils.disc_labels import decode_disc_labels

# Load the trained model
model = load_model()

# Sample email text
email_text = """
Hi team,

I need the quarterly report finalized by end of day.
Please prioritize this and let me know immediately if there are any blockers.

Thanks,
John
"""

# Process and predict
cleaned_text = clean_msg_body(email_text)
embeddings = get_bert_embeddings([cleaned_text])
prediction = model.predict(embeddings)
disc_labels = decode_disc_labels(prediction)

print(f"Predicted DISC style(s): {disc_labels[0]}")
# Output: Predicted DISC style(s): ('D',)
```

---

## Project Structure

```
disc-predictor-for-business/
├── app.py                      # Streamlit web application entry point
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── LICENSE                     # Apache 2.0 License
│
├── models/
│   └── xgb_model.joblib        # Trained XGBoost model
│
├── utils/
│   ├── preprocess.py           # Email text cleaning (20+ regex patterns)
│   ├── bert_embedder.py        # BERT embedding generation
│   ├── model_loader.py         # Model loading utilities
│   └── disc_labels.py          # Label encoding/decoding
│
├── model_data/
│   ├── X_train.npy             # Training embeddings
│   ├── X_test.npy              # Test embeddings
│   ├── Y_train.npy             # Training labels
│   └── Y_test.npy              # Test labels
│
├── data/
│   └── emails.csv              # Source email data
│
├── notebooks/                  # Data pipeline notebooks
│   ├── 01_data_transformation.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_postprocess_after_manual_label.ipynb
│   ├── 04_data_preprocess.ipynb
│   ├── 05_generate_BERT_embeddings.ipynb
│   ├── 06_train_and_evaluate_classifiers.ipynb
│   ├── 07_bootstrapping_unlabeled_emails.ipynb
│   └── 08_final_model_training.ipynb
│
├── tfhub_modules/              # Cached BERT models for offline use
│   ├── bert_preprocess/
│   └── bert_encoder/
│
└── tests/
    └── test_model_output.py    # Unit tests for prediction pipeline
```

---

## Model Performance

The XGBoost multi-label classifier achieves strong performance across all DISC categories:

| Metric | D | I | S | C |
|--------|---|---|---|---|
| Precision | 0.88 | 0.89 | 0.90 | 0.91 |
| AUC-ROC | 0.96+ | 0.96+ | 0.96+ | 0.96+ |

### Training Details

| Parameter | Value |
|-----------|-------|
| Training samples | 49,643 |
| Test samples | 12,411 |
| Embedding dimensions | 768 |
| Model type | XGBoost (MultiOutputClassifier) |
| BERT variant | bert_en_uncased_L-12_H-768_A-12 |

---

## Data Pipeline

The project uses an 8-stage data pipeline, implemented as Jupyter notebooks:

### Stage 1: Data Extraction
Extract emails from the Enron Email Corpus (517,401 emails).

### Stage 2: Data Cleaning
Apply text preprocessing with 20+ regex patterns to remove:
- Email headers (To, From, Subject, Cc, Bcc)
- Signatures and sign-offs
- URLs and file attachments
- Phone numbers and email addresses
- Dates and times
- HTML content

### Stage 3: Manual Labeling
Approximately 100 emails manually labeled by domain experts.

### Stage 4: Rule-Based Labeling
DISC keyword matching applied to approximately 2,400 additional emails.

### Stage 5: BERT Embedding Generation
Generate 768-dimensional embeddings using TensorFlow Hub BERT encoder.

### Stage 6: Classifier Training and Evaluation
Train and compare multiple classifiers (Logistic Regression, SVM, XGBoost).

### Stage 7: Pseudo-Labeling
Bootstrap remaining unlabeled emails using Logistic Regression predictions.

### Stage 8: Final Model Training
Train production XGBoost model on the complete labeled dataset.

---

## API Reference

### Preprocessing Module

**`utils/preprocess.py`**

```python
clean_msg_body(msg_body: str) -> str
```

Cleans raw email text by removing headers, signatures, URLs, and normalizing whitespace.

**Parameters:**
- `msg_body`: Raw email text string

**Returns:**
- Cleaned text string suitable for embedding generation

### Embedding Module

**`utils/bert_embedder.py`**

```python
get_bert_embeddings(texts: list) -> np.ndarray
```

Generates BERT embeddings for a list of text strings.

**Parameters:**
- `texts`: List of text strings to embed

**Returns:**
- NumPy array of shape (N, 768) containing embeddings

### Model Module

**`utils/model_loader.py`**

```python
load_model(path: str = None) -> sklearn.multioutput.MultiOutputClassifier
```

Loads the trained XGBoost model from disk.

**Parameters:**
- `path`: Optional custom path to model file (defaults to `models/xgb_model.joblib`)

**Returns:**
- Loaded scikit-learn MultiOutputClassifier

### Label Module

**`utils/disc_labels.py`**

```python
decode_disc_labels(encoded_disc_labels: np.ndarray) -> tuple
```

Converts binary prediction arrays to DISC label tuples.

**Parameters:**
- `encoded_disc_labels`: Binary array of shape (N, 4)

**Returns:**
- Tuple of DISC labels (e.g., `('D', 'I')`, `('S',)`)

---

## Testing

Run the test suite:

```bash
python -m unittest tests/test_model_output.py
```

The tests verify the complete prediction pipeline from raw text input to DISC label output.

---

## Known Limitations

1. **Label Quality**: Keyword-based labels may introduce bias and reduce precision compared to fully manual labeling.

2. **Short/Formal Emails**: Brief or highly formal messages often lack sufficient DISC signals for reliable classification.

3. **Pseudo-Label Confidence**: Pseudo-labels generated via Logistic Regression may include false positives, particularly in low-confidence cases.

4. **Docker Image Size**: The current Docker image is approximately 3.6 GB due to TensorFlow dependencies, which may be prohibitive for lightweight deployments.

5. **BERT Latency**: Real-time embedding generation adds latency; consider batch processing for high-throughput applications.

---

## Future Work

- [ ] Increase manual labeling volume to improve label quality
- [ ] Implement confidence thresholds for pseudo-labeling
- [ ] Add PostgreSQL/MySQL support for production storage
- [ ] Optimize Docker image size using TensorFlow Lite or ONNX
- [ ] Add batch prediction API endpoint
- [ ] Implement model versioning and A/B testing
- [ ] Create fine-tuned BERT model for email domain

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Dataset: [Enron Email Corpus](https://www.cs.cmu.edu/~enron/)
- BERT Models: [TensorFlow Hub](https://tfhub.dev/)
- DISC Framework: Based on the work of William Moulton Marston
