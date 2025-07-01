# DISC Predictor for Business Communication

This project predicts the DISC personality style of an email sender using natural language processing and machine learning. Understanding these styles helps businesses and individuals improve communication in the workplace.

> D – Dominance (Red)  
> I – Influence (Yellow)  
> S – Steadiness (Green)  
> C – Conscientiousness (Blue)

---

## Project Goals

- Classify business emails into DISC personality styles.
- Develop a working DISC classification system for real-world business communication.

---

## Features

- Preprocessing pipeline for noisy corporate emails (headers, forwards, signatures).
- BERT-based embeddings for semantic representation.
- Hybrid labeling strategy: manual, rule-based, and pseudo-labeling.
- Final model: XGBoost wrapped in `MultiOutputClassifier`.
- Streamlit MVP for predictions.
- Dockerized environment with basic testing.

---

## Data Overview

- **Dataset**: Enron Email Corpus (517,401 emails)
- **Subset used**: ~65,000 cleaned and filtered emails
- **Labeling Approach**:
  - ~100 emails manually labeled
  - ~2400 labeled with DISC keyword matching
  - Remaining pseudo-labeled via Logistic Regression
  - Iterative improvement planned for label accuracy

---

## Architecture

| Component     | Technology                     |
|---------------|--------------------------------|
| Language      | Python                         |
| Embeddings    | BERT (TensorFlow Hub)          |
| Classifier    | XGBoost (MultiOutputClassifier)|
| Interface     | Streamlit                      |
| Container     | Docker                         |
| Testing       | unittest                       |

---

## Known Limitations

- Keyword-based labels introduce bias and reduce label precision.
- Many emails lack clear DISC signals, especially in short or formal messages.
- Pseudo-labels are prone to false positives in low-confidence cases.
- Docker image size (~3.6 GB) is currently too large for lightweight deployment.

---

## Future Work

- Increase manual labeling volume to replace rule-based labels.
- Introduce confidence thresholds for pseudo-labeling.
- Replace SQLite with PostgreSQL or MySQL for production storage.
- Optimize Docker image for production use.

