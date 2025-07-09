from flask import Flask, request, render_template
import os
from pdfminer.high_level import extract_text
import docx2txt
import tempfile
import spacy
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)


nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight



def extract_resume_text(uploaded_file):
    _, ext = os.path.splitext(uploaded_file.filename)
    
    if ext.lower() == '.pdf':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            uploaded_file.save(tmp.name)
            text = extract_text(tmp.name)
        os.remove(tmp.name)
        return text

    elif ext.lower() == '.docx':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
            uploaded_file.save(tmp.name)
            text = docx2txt.process(tmp.name)
        os.remove(tmp.name)
        return text

    else:
        return "Unsupported file format"

def extract_keywords(text):
    doc = nlp(text)
    keywords = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
            keywords.append(token.text.lower())
    return list(set(keywords)) 

def get_similarity(resume_text, job_text):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(resume_embedding, job_embedding)
    return round(float(score[0][0]) * 100, 2) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    resume_file = request.files['resume']
    job_description = request.form['job_desc']

    resume_text = extract_resume_text(resume_file)
    job_text = job_description

    resume_keywords = set(extract_keywords(resume_text))
    job_keywords = set(extract_keywords(job_text))

    semantic_score = get_similarity(resume_text, job_text)

    if job_keywords:
        matched = resume_keywords & job_keywords
        skill_score = min(len(matched) / len(job_keywords) * 100, 100)
    else:
        skill_score = 0

    final_score = round((0.4 * skill_score) + (0.6 * semantic_score), 2)
    missing_keywords = sorted(job_keywords - resume_keywords)

    return render_template(
        'results.html',
        final_score=final_score,
        skill_score=round(skill_score, 2),
        semantic_score=round(semantic_score, 2),
        missing_keywords=missing_keywords
    )

if __name__ == '__main__':
    app.run(debug=True)
