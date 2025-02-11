streamlit run mcq_generator.py
import streamlit as st
import pdfplumber
import nltk
import random
import pandas as pd
from transformers import pipeline

# Download necessary NLP resources
nltk.download('punkt')

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

# Function to extract text from a TXT file
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to generate MCQs
def generate_mcqs(text):
    # Summarize text (optional)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text[:1024], max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    # Generate questions
    qa_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
    questions = qa_pipeline(summary, max_length=100, num_return_sequences=5, do_sample=True)

    # Function to create MCQs with multiple options
    mcqs = []
    for q in questions:
        question_text = q['generated_text']
        correct_answer = "Correct Answer"  # Placeholder (improve with NLP)
        options = [correct_answer] + random.sample(["Option A", "Option B", "Option C"], 3)
        random.shuffle(options)
        mcqs.append({
            "Question": question_text,
            "Option 1": options[0],
            "Option 2": options[1],
            "Option 3": options[2],
            "Option 4": options[3],
            "Correct Answer": correct_answer
        })
    
    return mcqs

# Streamlit UI
st.title("📄 MCQ Generator from Text/PDF")
st.write("Upload a text or PDF file to generate multiple-choice questions.")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

if uploaded_file:
    file_type = uploaded_file.type
    text = ""

    if file_type == "text/plain":
        text = extract_text_from_txt(uploaded_file)
    elif file_type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)

    if text:
        st.success("✅ File uploaded successfully!")
        mcq_list = generate_mcqs(text)

        # Display MCQs
        st.subheader("Generated MCQs")
        for idx, mcq in enumerate(mcq_list, start=1):
            st.write(f"**Q{idx}:** {mcq['Question']}")
            st.write(f" 1️⃣ {mcq['Option 1']}")
            st.write(f" 2️⃣ {mcq['Option 2']}")
            st.write(f" 3️⃣ {mcq['Option 3']}")
            st.write(f" 4️⃣ {mcq['Option 4']}")
            st.write(f"✅ **Correct Answer:** {mcq['Correct Answer']}")
            st.write("---")

        # Convert MCQs to DataFrame
        df = pd.DataFrame(mcq_list)

        # Save as CSV
        csv_filename = "generated_mcqs.csv"
        df.to_csv(csv_filename, index=False)
        st.download_button(label="📥 Download MCQs as CSV", data=df.to_csv(index=False), file_name=csv_filename, mime="text/csv")
    else:
        st.error("❌ No text extracted from the file!")
