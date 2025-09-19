import pytesseract
from PIL import Image
import pdf2image
import requests
import json
import os
import pandas as pd
from pathlib import Path
import time
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Directory and output file
PDF_DIRECTORY = "invoices"
OUTPUT_DIR = "output"
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "invoices.xlsx")
DEBUG_TEXT_DIR = "debug_text"

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MAX_RETRIES = 3
RETRY_DELAY = 2

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF invoice by converting pages to images."""
    try:
        images = pdf2image.convert_from_path(pdf_path)
        full_text = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang='eng+fra+ara')
            full_text += f"--- Page {i+1} ---\n{text}\n"
        os.makedirs(DEBUG_TEXT_DIR, exist_ok=True)
        debug_file = os.path.join(DEBUG_TEXT_DIR, f"{Path(pdf_path).stem}_extracted.txt")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        logging.info(f"Extracted text saved to {debug_file}")
        return full_text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return None

def process_all_invoices(invoice_texts):
    """Send all extracted texts to Gemma 3 via OpenRouter API in a single call."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    text_sections = []
    for pdf_name, text in invoice_texts:
        text_sections.append(f"Invoice: {pdf_name}\n{text}\n")
    combined_text = "\n".join(text_sections)

    prompt = (
        "You are an expert in extracting and translating invoice details for the MENA region. "
        "Given the following raw texts extracted from multiple invoices, extract the following details for each invoice: "
        "invoice number, date, total amount, vendor name, vendor address, and tax identification number. "
        "The texts may be in English, French, or Arabic. "
        "Return a list of JSON objects, each with field names in Arabic: "
        "'رقم الفاتورة', 'التاريخ', 'المبلغ الإجمالي', 'اسم المورد', 'عنوان المورد', 'رقم التعريفي الجبائي'. "
        "The field 'رقم الفاتورة' may be stated as 'Invoice Number', 'Numéro de la facture', 'فاتورة رقم', or similar variations. "
        "For the tax identification number, look for terms like 'Tax ID', 'VAT Number', 'Numéro d’identification fiscale', "
        "'Matricule Fiscal' or 'MF' or 'Registre de commerce' (Tunisia), 'Identifiant Fiscal Commun (IFC)' (Morocco), "
        "'Tax Registration Number' (Egypt), or 'رقم التعريفي الجبائي' in Arabic, depending on the MENA country. "
        "Note: Fields like 'رقم الفاتورة', 'التاريخ', 'المبلغ الإجمالي', and 'رقم التعريفي الجبائي' may contain Arabic numerals "
        "(٠=0, ١=1, ٢=2, ٣=3, ٤=4, ٥=5, ٦=6, ٧=7, ٨=8, ٩=9, e.g., ١٠=10). Convert these to Western numerals (0-9) in the output, "
        "except for 'رقم التعريفي الجبائي', which should preserve the original value if it’s a number or code. "
        "If a field is in English or French, translate it to Arabic for 'اسم المورد' and 'عنوان المورد'. "
        "If any information is missing or unclear, indicate it as 'غير موجود'. "
        "Include the invoice filename as 'اسم الملف' in each JSON object. "
        "Return a valid JSON array of objects without Markdown code fences (```json or ```), backticks, or any non-JSON text. "
        f"{combined_text}\n\n"
        "Example output: "
        "[{\"اسم الملف\":\"invoice1.pdf\",\"رقم الفاتورة\":\"123\",\"التاريخ\":\"2023-10-01\","
        "\"المبلغ الإجمالي\":\"1000\",\"اسم المورد\":\"شركة مثال\",\"عنوان المورد\":\"شارع مثال، تونس\","
        "\"رقم التعريفي الجبائي\":\"١٢٣٤٥٦٧٨٩\"}]"
    )

    payload = {
        "model": "google/gemma-3-27b-it:free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for extracting and translating invoice details."},
            {"role": "user", "content": prompt}
        ]
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            debug_response_file = os.path.join(DEBUG_TEXT_DIR, "all_invoices_response.json")
            os.makedirs(DEBUG_TEXT_DIR, exist_ok=True)
            with open(debug_response_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logging.info(f"API response saved to {debug_response_file}")
            
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            if not content:
                logging.error("Empty content in API response")
                continue
            content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
            content = content.strip('`')
            try:
                extracted_data = json.loads(content)
                if not isinstance(extracted_data, list):
                    logging.error("API response is not a JSON array")
                    continue
                return extracted_data
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}")
                logging.error(f"Raw content: {content}")
                continue
        except requests.RequestException as e:
            logging.error(f"API request error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            continue
    logging.error(f"Failed to process invoices after {MAX_RETRIES} attempts")
    return None

def main():
    if not OPENROUTER_API_KEY:
        logging.error("OPENROUTER_API_KEY environment variable not set")
        return

    if not os.path.exists(PDF_DIRECTORY):
        logging.error(f"Directory {PDF_DIRECTORY} does not exist")
        return

    invoice_texts = []
    for pdf_file in Path(PDF_DIRECTORY).glob("*.pdf"):
        logging.info(f"Extracting text from {pdf_file.name}...")
        extracted_text = extract_text_from_pdf(pdf_file)
        if not extracted_text:
            logging.error(f"Failed to extract text from {pdf_file.name}")
            continue
        invoice_texts.append((pdf_file.name, extracted_text))

    if not invoice_texts:
        logging.error("No invoice texts were extracted successfully")
        return

    logging.info("Processing all invoices with a single API call...")
    results = process_all_invoices(invoice_texts)
    if not results:
        logging.error("Failed to process invoice details")
        return

    if results:
        df = pd.DataFrame(results, columns=[
            'اسم الملف', 'رقم الفاتورة', 'التاريخ', 'المبلغ الإجمالي', 'اسم المورد', 'عنوان المورد', 'رقم التعريفي الجبائي'
        ])
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if os.path.exists(OUTPUT_EXCEL):
            existing_df = pd.read_excel(OUTPUT_EXCEL, engine='openpyxl')
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_excel(OUTPUT_EXCEL, index=False, engine='openpyxl')
        logging.info(f"Results appended to {OUTPUT_EXCEL}")
    else:
        logging.error("No invoices were processed successfully")