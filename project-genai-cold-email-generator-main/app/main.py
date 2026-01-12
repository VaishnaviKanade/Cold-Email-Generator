import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from utils import clean_text, read_resume

def create_streamlit_app(llm):
    st.title("📧 Resume-Based Cold Email Generator")

    job_url = st.text_input("Enter Job URL")
    resume_file = st.file_uploader(
        "Upload Resume (PDF or DOCX)",
        type=["pdf", "docx"]
    )

    tone = st.selectbox(
        "Select Email Tone",
        ["Formal", "Friendly", "Confident"]
    )

    length = st.selectbox(
        "Select Email Length",
        ["Short (4–5 lines)", "Medium (7–8 lines)", "Long (10–12 lines)"]
    )

    if st.button("Generate Cold Email"):
        if not job_url or not resume_file:
            st.warning("Please upload resume and enter job URL")
            return

        try:
            job_loader = WebBaseLoader([job_url])
            job_text = clean_text(job_loader.load()[0].page_content)

            resume_text = clean_text(read_resume(resume_file))

            jobs = llm.extract_jobs(job_text)

            if not isinstance(jobs, list):
                jobs = [jobs]

            for job in jobs:
                email = llm.write_mail(job, resume_text, tone, length)
                #st.code(email, language="markdown")
                st.text_area(label="Generated Email",value=email,height=450,)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    #st.set_page_config(page_title="Cold Email Generator", layout="wide")
    st.set_page_config(page_title="Cold Email Generator", layout="centered")

    create_streamlit_app(Chain())

