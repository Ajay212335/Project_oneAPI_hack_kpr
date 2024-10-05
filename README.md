Sensitive Data Masking, Resume Parsing, and GAN-based Synthetic Data Generation

## Overview
This project is a Streamlit application that enables users to upload various document types (PDF, TXT, DOCX, JPG, JPEG, PNG) and perform the following functions:

- **Text Extraction**: Extracts text from uploaded documents.
- **Document Type Detection**: Identifies if the document is a resume, Aadhar card, or PAN card.
- **Data Extraction**: Extracts structured data from the identified document type.
- **Sensitive Data Masking**: Masks sensitive information such as names, emails, and phone numbers.
- **Synthetic Data Generation**: Utilizes a GAN-based approach to generate synthetic data based on the extracted information.

## Features
- Support for multiple document formats.
- Natural Language Processing (NLP) for structured data extraction.
- Optical Character Recognition (OCR) for text extraction from images.
- Generation of synthetic data with customizable fields.

## Requirements
- Python 3.x
- Streamlit
- Pandas
- PyPDF2
- Pillow
- python-docx
- pytesseract
- TensorFlow
- spaCy

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. Create a virtual environment (optional):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR as per the instructions for your OS from the [Tesseract GitHub repository](https://github.com/tesseract-ocr/tesseract).

5. Download the spaCy English model:
   ```
   python -m spacy download en_core_web_sm
   ```

## Usage
1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your browser and go to `http://localhost:8501`.

3. Upload your document to process it for data extraction, masking, and synthetic data generation.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Streamlit for building interactive web applications.
- TensorFlow for machine learning capabilities.
- spaCy for Natural Language Processing.
- Tesseract OCR for text extraction from images.

--- 
