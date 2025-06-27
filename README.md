# 🎓 Educational Feedback Generation System

Leverage the power of AI to deliver meaningful, personalized feedback to students. This project combines classical machine learning with state-of-the-art Large Language Models (LLMs) to analyze student performance and generate actionable, supportive educational guidance.

---

## 🚀 Project Overview

This system is built as a robust, modular pipeline:

1. **Data Preprocessing**
   Cleans, encodes, and transforms raw educational data (e.g., ASSISTments, Portuguese Student dataset, etc.) into structured, model-ready features.

2. **Predictive Model Training**
   Trains a classical ML model (e.g., Random Forest) to predict student performance and identify learning gaps.

3. **LLM Fine-Tuning**
   Fine-tunes a Mistral-7B LLM using QLoRA on real educational dialogue data—optimized for educational feedback tasks and memory-efficient inference.

4. **Feedback Generation**
   Integrates ML predictions and LLM insights to produce detailed, context-aware, and motivational feedback for each student.

5. **Evaluation**
   Rigorously assesses both the predictive accuracy and the quality of generated feedback using automated metrics and, optionally, human review.

---

## 📚 Dataset Information

* **ASSISTments Dataset:**
  Large-scale student interaction logs from an intelligent tutoring system.

* **Education Dialogue Dataset:**
  Authentic teacher-student conversations for realistic LLM fine-tuning.

* **Portuguese Student Dataset:**
  Demographic, social, and academic records for grade prediction and feedback.

---

## 🗂️ Project Structure

```
project-root/
│
├── data/
│   ├── raw/               # Original datasets
│   └── processed/         # Cleaned and feature-engineered data
│
├── models/
│   ├── predictive/        # Trained ML models
│   └── llm/               # Fine-tuned LLM adapters
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_predictive_model.py
│   ├── fine_tune_llm.ipynb           # ran in Kaggle GPU
│   ├── feedback_generator.py
│   ├── pipeline_integration.py
│   └── evaluation.py
│
├── outputs/
│   ├── batch_feedback.json           # Batch feedback results
│   ├── single_student_feedback.json  # Individual feedback
│   └── evaluation_report.json        # System evaluation report
│
├── visualizations/                   # Charts and analytics
│
└── requirements.txt                  # Python dependencies
```

---

## ⚡ Installation

**Clone the repository**

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Usage

### Run the Full Pipeline

```bash
python src/pipeline_integration.py --data student-merged.csv
```

### Run Individual Components

**Data Preprocessing:**

```bash
python src/pipeline_integration.py --data student-merged.csv --preprocess-only
```

**Model Training:**

```bash
python src/pipeline_integration.py --train-only
```

**Feedback Generation (for a specific student):**

```bash
python src/pipeline_integration.py --feedback-only --student 123
```

**Evaluation:**

```bash
python src/evaluation.py
```

---

## 🧠 LLM Fine-Tuning (Kaggle Recommended)

* Upload `src/fine_tune_llm.py` and the Education Dialogue Dataset to Kaggle.
* Run the script in a GPU-accelerated Kaggle notebook.
* Download the fine-tuned model adapters for local deployment.

---

## 📊 Evaluation Metrics

**Predictive Model:**

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score

**Feedback Quality:**

* BLEU Score
* ROUGE Score
* Human Ratings (optional)

---

## 💡 Notes & Best Practices

* Handles large datasets efficiently with chunked preprocessing.
* LLM fine-tuning uses 4-bit quantization (QLoRA) for memory savings.
* Feedback generation gracefully falls back to template-based output if LLM is unavailable.
* Modular design allows easy extension to new datasets or feedback styles.

---

## 📄 License

This project is for educational and research purposes only.
Datasets used are subject to their original licenses and terms of use.

---

Empower every student with actionable, AI-powered feedback.
Built with open-source tools, real educational data, and a passion for learning.
