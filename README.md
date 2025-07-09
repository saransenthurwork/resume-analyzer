# 🧠 AI Resume Analyzer

A smart web app that scores how well a resume matches a job description using NLP and Deep Learning.

---

## 🔍 Features

- Upload PDF/DOCX resume
- Paste job description
- Analyze skill match and semantic similarity
- Shows Job Fit Score + missing keywords
- Clean responsive UI

---

## 🛠️ Tech Stack

- Flask, Python
- spaCy, BERT (Sentence Transformers)
- Bootstrap 5 (Frontend)

---

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/resume-analyzer.git
cd resume-analyzer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
