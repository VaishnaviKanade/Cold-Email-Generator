import re
from pypdf import PdfReader
from docx import Document

def clean_text(text):
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ,.]+', ' ', text)
    return ' '.join(text.split())

def read_resume(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join(page.extract_text() or "" for page in reader.pages)

    elif file.name.endswith(".docx"):
        doc = Document(file)
        return " ".join(p.text for p in doc.paragraphs)

    else:
        raise ValueError("Unsupported file format")
