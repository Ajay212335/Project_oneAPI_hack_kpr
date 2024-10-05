import streamlit as st
import pandas as pd
import re
from PyPDF2 import PdfReader
from PIL import Image
import docx
import pytesseract
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import spacy

# Set up Tesseract for OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
nlp = spacy.load("en_core_web_sm")

# Streamlit app configuration
st.set_page_config(page_title="Sensitive Data Masking, Resume Parsing, and GAN-based Synthetic Data Generation", layout="wide")
st.title("Sensitive Data Masking, Resume Parsing, and GAN-based Synthetic Data Generation")

# File uploader
uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt", "docx", "jpg", "jpeg", "png"])

def parse_pdf(file):
    """Extract text from a PDF document."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def parse_txt(file):
    """Extract text from a TXT file."""
    return file.read().decode('utf-8')

def parse_docx(file):
    """Extract text from a DOCX file."""
    doc = docx.Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def parse_image(file):
    """Extract text from an image using OCR."""
    image = Image.open(file)
    gray_image = image.convert('L')  
    text = pytesseract.image_to_string(gray_image)
    return text

def detect_document_type(text):
    """Detect the type of document based on keywords."""
    text_lower = text.lower()
    if "resume" in text_lower or "career objective" in text_lower:
        return "Resume"
    elif "aadhar" in text_lower or "aadhaar" in text_lower:
        return "Aadhar Card"
    elif "pan" in text_lower or "permanent account number" in text_lower:
        return "PAN Card"
    else:
        return "Unknown"

def extract_with_nlp(text):
    """Use NLP to extract structured data."""
    doc = nlp(text)
    data = {
        'Name': None,
        'Email': None,
        'Phone': None,
        'Education': [],
        'Skills': [],
        'Hobbies': []
    }

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            data['Name'] = ent.text
        elif ent.label_ == 'EMAIL':
            data['Email'] = ent.text
        elif ent.label_ == 'PHONE':
            data['Phone'] = ent.text

    return data

def extract_resume_data(full_text):
    """Extract structured data from a resume using regex and NLP."""
    data = extract_with_nlp(full_text)

    # Enhanced name extraction
    name_match = re.search(r'(?<!\w)([A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)+)(?<!\w)', full_text)
    if name_match:
        data['Name'] = name_match.group(0).strip()

    # Regex for education
    education_match = re.findall(r'\b(?:Bachelor|Master|B\.Sc|M\.Sc|Bachelors|Masters|BA|BS|MA|MS|MBA)[^\n]*', full_text, re.IGNORECASE)
    data['Education'] = education_match if education_match else []

    # Enhanced skills extraction
    skills_match = re.search(r'(SKILLS|TECHNICAL SKILLS)[\s:\-]*(.*?)(?:\n\n|\n[A-Z])', full_text, re.IGNORECASE)
    if skills_match:
        skills = re.split(r',|\n', skills_match.group(2).strip())
        data['Skills'] = [skill.strip() for skill in skills if skill.strip()]

    # Enhanced hobbies extraction
    hobbies_match = re.search(r'(HOBBIES|INTERESTS)[\s:\-]*(.*?)(?=\n\n|\n[A-Z]|$)', full_text, re.DOTALL | re.IGNORECASE)
    if hobbies_match:
        hobbies = re.split(r',|\n', hobbies_match.group(2).strip())
        data['Hobbies'] = [hobby.strip() for hobby in hobbies if hobby.strip()]

    return data

def extract_aadhar_data(full_text):
    """Extract personal details from Aadhar card."""
    data = {}
    aadhar_match = re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', full_text)
    if aadhar_match:
        data['Aadhar Number'] = aadhar_match.group(0).replace(' ', '')

    name_match = re.search(r'Name of the Student\s*[:\-]?\s*([A-Za-z\s]+)', full_text, re.IGNORECASE)
    if name_match:
        data['Name'] = name_match.group(1).strip()

    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', full_text)
    if email_match:
        data['Email'] = email_match.group(0).strip()

    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', full_text)
    if phone_match:
        data['Phone'] = phone_match.group(0).strip()

    return data

def extract_pan_data(full_text):
    """Extract personal details from PAN card."""
    data = {}
    pan_match = re.search(r'\b[A-Z]{5}\d{4}[A-Z]{1}\b', full_text)
    if pan_match:
        data['PAN Number'] = pan_match.group(0)

    name_match = re.search(r'Name\s*[:\-]?\s*([A-Za-z\s]+)', full_text, re.IGNORECASE)
    if name_match:
        data['Name'] = name_match.group(1).strip()

    father_match = re.search(r"Father's Name\s*[:\-]?\s*([A-Za-z\s]+)", full_text, re.IGNORECASE)
    if father_match:
        data['Father’s Name'] = father_match.group(1).strip()

    dob_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', full_text, re.IGNORECASE)
    if dob_match:
        data['Date of Birth'] = dob_match.group(1).strip()

    return data

def mask_sensitive_data(data):
    """Mask sensitive data in the extracted data dictionary."""
    masked_data = data.copy()
    for key in masked_data:
        if key.lower() in ['name', 'email', 'phone', 'aadhar number', 'pan number', 'father’s name', 'date of birth']:
            masked_data[key] = f"[{key.upper()} MASKED]"
    return masked_data

def make_generator_model():
    """Create the generator model for GAN."""
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, input_shape=(100,), activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(7, activation='tanh')) 
    return model

def make_discriminator_model():
    """Create the discriminator model for GAN."""
    model = tf.keras.Sequential()
    model.add(layers.Dense(512, input_shape=(7,), activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def generate_synthetic_data(data, doc_type):
    """Generate synthetic data using GAN based on document type."""
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    noise = np.random.normal(0, 1, (1, 100))  
    generated_data = generator(noise).numpy()

    if doc_type == "Resume":
        fields = ['Name', 'Email', 'Phone', 'Education', 'Skills', 'Projects', 'Hobbies']
    elif doc_type == "Aadhar Card":
        fields = ['Name', 'Email', 'Phone', 'Aadhar Number']
    elif doc_type == "PAN Card":
        fields = ['Name', 'Father’s Name', 'PAN Number', 'Date of Birth']
    else:
        fields = ['Field1', 'Field2', 'Field3'] 

    num_fields = min(len(fields), generated_data.shape[1])
    synthetic_data = generated_data[:, :num_fields]
    
    synthetic_df = pd.DataFrame(synthetic_data, columns=fields[:num_fields])
    
    return synthetic_df

# File processing logic
if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'pdf':
            extracted_text = parse_pdf(uploaded_file)
        elif file_extension == 'txt':
            extracted_text = parse_txt(uploaded_file)
        elif file_extension == 'docx':
            extracted_text = parse_docx(uploaded_file)
        elif file_extension in ['jpg', 'jpeg', 'png']:
            extracted_text = parse_image(uploaded_file)
        else:
            extracted_text = "Unsupported file type."
    except Exception as e:
        st.error(f"Error processing the file: {e}")
        extracted_text = ""
    
    if extracted_text and extracted_text != "Unsupported file type.":
        st.header("Extracted Text from Document")
        st.write(extracted_text)

        doc_type = detect_document_type(extracted_text)
        st.subheader(f"Detected Document Type: {doc_type}")

        extracted_data = {}

        if doc_type == "Resume":
            extracted_data = extract_resume_data(extracted_text)
        elif doc_type == "Aadhar Card":
            extracted_data = extract_aadhar_data(extracted_text)
        elif doc_type == "PAN Card":
            extracted_data = extract_pan_data(extracted_text)
        else:
            st.warning("Document type not recognized. No data extracted.")

        if extracted_data:
            st.header("Extracted Personal Data")
            st.json(extracted_data)

            masked_data = mask_sensitive_data(extracted_data)

            st.header("Masked Personal Data")
            st.json(masked_data)

            synthetic_df = generate_synthetic_data(extracted_data, doc_type)
            st.header("Synthetic Data (Generated)")
            st.dataframe(synthetic_df)

            st.json(synthetic_df.to_dict(orient='records'))
    elif extracted_text == "Unsupported file type.":
        st.warning("Unsupported file type. Please upload a PDF, TXT, DOCX, JPG, JPEG, or PNG file.")
    else:
        st.warning("No text extracted from the uploaded file.")
