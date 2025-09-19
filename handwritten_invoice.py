import os
import logging
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import pdf2image
import base64
import time
from mimetypes import guess_type
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
# Configure logging with detailed output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('invoice_processor_debug.log'),
        logging.StreamHandler()
    ]
)
load_dotenv()
# Directory and output file
OUTPUT_DIR = "output"
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "handwritten_invoices.xlsx")
DEBUG_TEXT_DIR = "debug_text"
DEBUG_IMAGE_DIR = "debug_images"
DEBUG_BASE64_DIR = "debug_base64"

# Open AI API configuration
OPENAI_API_KEY = OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY environment variable not set")
    raise ValueError("OPENAI_API_KEY environment variable is required")
client = OpenAI(api_key=OPENAI_API_KEY)
MAX_RETRIES = 3
RETRY_DELAY = 2


def is_image_file(file_path):
    """Determine if the file is an image based on its MIME type."""
    mime_type, _ = guess_type(file_path)
    is_image = mime_type and mime_type.startswith('image')
    logging.debug(f"File {file_path} MIME type: {mime_type}, is_image: {is_image}")
    return is_image


def convert_pdf_to_images(pdf_path):
    """Convert PDF to a list of PIL images."""
    try:
        os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
        images = pdf2image.convert_from_path(pdf_path)
        logging.info(f"Converted {pdf_path} to {len(images)} image(s)")

        # Save images for debugging
        for i, image in enumerate(images):
            debug_image_path = os.path.join(DEBUG_IMAGE_DIR, f"{Path(pdf_path).stem}_page_{i+1}.png")
            image.save(debug_image_path, "PNG")
            logging.debug(f"Saved debug image: {debug_image_path}")

        return images
    except Exception as e:
        logging.error(f"Error converting PDF {pdf_path}: {str(e)}", exc_info=True)
        return []


def image_to_base64(image, file_name="unknown", image_index=0):
    """Convert a PIL image to base64 string."""
    try:
        # Verify image is valid
        if not isinstance(image, Image.Image):
            logging.error(f"Invalid image object for {file_name} (image {image_index})")
            return None

        logging.debug(f"Image mode: {image.mode}, size: {image.size} for {file_name} (image {image_index})")

        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Validate base64 string
        if not base64_string:
            logging.error(f"Empty base64 string generated for {file_name} (image {image_index})")
            return None

        # Save base64 string for debugging
        os.makedirs(DEBUG_BASE64_DIR, exist_ok=True)
        debug_base64_file = os.path.join(DEBUG_BASE64_DIR, f"{Path(file_name).stem}_image_{image_index}_base64.txt")
        with open(debug_base64_file, 'w', encoding='utf-8') as f:
            f.write(base64_string)
        logging.debug(f"Saved base64 string to {debug_base64_file}, length: {len(base64_string)}")

        return base64_string
    except Exception as e:
        logging.error(
            f"Error converting image to base64 for {file_name} (image {image_index}): {str(e)}", exc_info=True)
        return None


def extract_text_from_image(image, image_index=0, file_name="unknown"):
    """Extract handwritten text from a PIL image using Open AI gpt-4.1-mini."""
    try:
        base64_image = image_to_base64(image, file_name, image_index)
        if not base64_image:
            logging.error(f"No base64 image generated for {file_name} (image {image_index})")
            return ""

        # Prompt based on Promptly Engineering guide
        prompt = (
            "You are an expert in recognizing handwritten text from invoices. "
            "Extract all text from the provided image, which is a handwritten invoice. "
            "The text may be in English, French, or Arabic, and may be noisy or difficult to read due to handwriting. "
            "Return the raw text as a string, preserving the original content as closely as possible. "
            "Do not interpret or summarize the text, only extract it verbatim."
        )

        logging.debug(
            f"Calling OpenAI API (model: gpt-4.1-mini) for text extraction from {file_name} (image {image_index})")

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "auto"}}
                        ]
                    }],
                    temperature=0.0,
                    max_tokens=1000
                )
                text = response.choices[0].message.content.strip()

                # Log extracted text
                logging.debug(f"Extracted text length from {file_name} (image {image_index}): {len(text)} characters")
                if not text:
                    logging.warning(f"Empty text extracted from {file_name} (image {image_index})")

                # Save extracted text for debugging
                os.makedirs(DEBUG_TEXT_DIR, exist_ok=True)
                debug_text_file = os.path.join(DEBUG_TEXT_DIR, f"{Path(file_name).stem}_image_{image_index}_raw.txt")
                with open(debug_text_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                logging.debug(f"Saved raw extracted text to {debug_text_file}")

                return text
            except OpenAIError as e:
                logging.error(
                    f"OpenAI API error for {file_name} (image {image_index}, attempt {attempt+1}/{MAX_RETRIES}): {str(e)}", exc_info=True)
                if attempt < MAX_RETRIES - 1:
                    logging.debug(f"Retrying after {RETRY_DELAY} seconds")
                    time.sleep(RETRY_DELAY)
                continue
        logging.error(f"Failed to extract text from {file_name} (image {image_index}) after {MAX_RETRIES} attempts")
        return ""
    except Exception as e:
        logging.error(f"Error extracting text from {file_name} (image {image_index}): {str(e)}", exc_info=True)
        return ""


def extract_text_from_file(file_path):
    """Extract text from a PDF or image file, processing each PDF page as a separate image."""
    try:
        file_name = Path(file_path).name
        full_text = ""
        os.makedirs(DEBUG_TEXT_DIR, exist_ok=True)
        debug_file = os.path.join(DEBUG_TEXT_DIR, f"{Path(file_path).stem}_extracted.txt")

        logging.debug(f"Processing file: {file_name}")

        if is_image_file(file_path):
            logging.info(f"Processing image file: {file_name}")
            image = Image.open(file_path).convert("RGB")

            # Save input image for debugging
            debug_image_path = os.path.join(DEBUG_IMAGE_DIR, f"{Path(file_name).stem}.png")
            os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
            image.save(debug_image_path, "PNG")
            logging.debug(f"Saved input image for debugging: {debug_image_path}")

            text = extract_text_from_image(image, image_index=0, file_name=file_name)
            full_text = text
        else:
            logging.info(f"Processing PDF file: {file_name}")
            images = convert_pdf_to_images(file_path)
            if not images:
                logging.error(f"No images generated from PDF {file_name}")
                return None

            for i, image in enumerate(images):
                text = extract_text_from_image(image, image_index=i + 1, file_name=file_name)
                full_text += f"--- Page {i+1} ---\n{text}\n"

        if not full_text.strip():
            logging.warning(f"No text extracted from {file_name}")

        # Save combined extracted text
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        logging.info(f"Extracted text saved to {debug_file}")

        return full_text.strip()
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        return None


def process_all_invoices(invoice_texts, output_excel=None):
    """Send extracted texts to GPT-4 via Open AI API and save results to Excel."""
    try:
        text_sections = []
        for file_name, text in invoice_texts:
            logging.debug(f"Processing invoice text for {file_name}, text length: {len(text)}")
            text_sections.append(f"Invoice: {file_name}\n{text}\n")
        combined_text = "\n".join(text_sections)

        if not combined_text.strip():
            logging.error("No text provided for field extraction")
            return None

        logging.debug(f"Combined text length for API call: {len(combined_text)}")

        prompt = (
            "You are an expert in extracting and translating invoice details for the MENA region. "
            "Given the following raw texts extracted from multiple handwritten invoices, extract the specified details for each invoice. "
            "The texts may be in English, French, or Arabic, and may be noisy due to handwritten text recognition. "
            "For each invoice, identify the table containing quantities, designations, unit prices, and amounts (e.g., 'Qté | Désignation | P.U | Montant'). "
            "Create a separate JSON object for each row in this table, repeating the non-table fields (invoice number, date, etc.) for each item. "
            "If the table is not explicitly formatted but contains itemized entries (e.g., 'Divers Transport | 3464,00', 'TVA 7% | 661,48'), treat each entry as a separate row. "
            "Return a valid JSON array of objects, each with field names in Arabic: "
            "'اسم الملف', 'رقم الفاتورة', 'التاريخ', 'رقم التعريفي الجبائي', 'بذمة', 'الكمية', 'البيانات', 'ثمن القطعة', 'القيمة', 'المجموع', 'اوقفت هذه الفاتورة بقيمة قدره', 'رقم الهاتف', 'اسم الشركة'. "
            "Handle the following variations and instructions:\n"
            "- 'رقم الفاتورة': May appear as 'Facture n°', 'Invoice Number', 'قائمة', 'فاتورة رقم', or similar (e.g., 'N° 53').\n"
            "- 'التاريخ': May appear as 'le', 'Date', 'Date de facture', 'التاريخ', or similar (e.g., 'le 30/04/2025').\n"
            "- 'رقم التعريفي الجبائي': May appear as 'MF', 'Tax ID', 'VAT Number', 'Numéro d’identification fiscale', 'Matricule Fiscal', 'Registre de commerce' (Tunisia), 'Identifiant Fiscal Commun (IFC)' (Morocco), 'Tax Registration Number' (Egypt), or 'رقم التعريفي الجبائي'; preserve the original value (e.g., '97887r').\n"
            "- 'بذمة': May appear as 'Doit (client)', 'Billed To', 'Facturé à', 'إلى', or similar (e.g., 'Doit : Vectorys Tunisie'); translate to Arabic.\n"
            "- 'الكمية': May appear as 'Qté', 'Quantity', 'Quantité', or similar; if missing, use 'غير موجود'.\n"
            "- 'البيانات': May appear as 'Désignation', 'Designation', 'Description', or similar; includes items like 'Divers Transport', 'TVA 7%', 'Timbre'.\n"
            "- 'ثمن القطعة': May appear as 'P.U', 'Unit Price', 'Prix unitaire', or similar; if missing, use 'غير موجود'.\n"
            "- 'القيمة': May appear as 'Montant', 'Amount', or similar; corresponds to each item’s amount (e.g., '3464,00', '661,48', '1,000').\n"
            "- 'المجموع': May appear as 'Total', 'Montant total', 'المجموع', or similar (e.g., 'TOTAL 40 124,48').\n"
            "- 'اوقفت هذه الفاتورة بقيمة قدره': May appear as 'Arrêter la présente facture à la somme de', 'Total in words', or similar; include full text (e.g., 'Dix Mille Cent vingt 4 sept 0 dinars, y 80 Millimes').\n"
            "- 'رقم الهاتف': May appear as 'tél', 'Phone', 'Téléphone', 'رقم الهاتف', or similar; include all numbers if multiple (e.g., '28 017 108 - 20.42 519').\n"
            "- 'اسم الشركة': May appear as the vendor or supplier name (e.g., 'HAMMAMI ZIAD Transporteur', 'Supplier', 'Fournisseur'); include full text.\n"
            "Convert Arabic numerals (٠=0, ١=1, ٢=2, ٣=3, ٤=4, ٥=5, ٦=6, ٧=7, ٨=8, ٩=9) to Western numerals (0-9) for 'رقم الفاتورة', 'التاريخ', 'المجموع', 'الكمية', 'ثمن القطعة', and 'القيمة', but preserve the original value for 'رقم التعريفي الجبائي'. "
            "If a field is in English or French, translate it to Arabic for 'بذمة'. "
            "If any information is missing or unclear, indicate it as 'غير موجود'. "
            "Include the invoice filename as 'اسم الملف' in each JSON object. "
            "Ensure the output is a valid JSON array of objects (e.g., [{...}, {...}]), even for a single invoice. "
            "Wrap the JSON output in triple backticks with 'json' identifier (```json\n[...]\n```). "
            "Example for a table with two items:\n"
            "Input: Qté | Désignation | P.U | Montant\n1 | Item1 | 100 | 100\n2 | Item2 | 200 | 400\n"
            "Output: ```json\n[\n"
            "  {\"اسم الملف\": \"file.jpg\", \"رقم الفاتورة\": \"123\", ..., \"الكمية\": \"1\", \"البيانات\": \"Item1\", \"ثمن القطعة\": \"100\", \"القيمة\": \"100\", ...},\n"
            "  {\"اسم الملف\": \"file.jpg\", \"رقم الفاتورة\": \"123\", ..., \"الكمية\": \"2\", \"البيانات\": \"Item2\", \"ثمن القطعة\": \"200\", \"القيمة\": \"400\", ...}\n"
            "]\n```\n"
            f"\n\n{combined_text}"
        )

        for attempt in range(MAX_RETRIES):
            try:
                logging.debug(
                    f"Calling OpenAI API (model: gpt-4) for field extraction (attempt {attempt+1}/{MAX_RETRIES})")

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for extracting and translating invoice details."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=4000
                )
                content = response.choices[0].message.content.strip()

                if not content:
                    logging.error("Empty content in API response")
                    continue

                # Log raw response for debugging
                debug_response_file = os.path.join(DEBUG_TEXT_DIR, f"all_invoices_response_attempt_{attempt+1}.txt")
                os.makedirs(DEBUG_TEXT_DIR, exist_ok=True)
                with open(debug_response_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                logging.info(f"Raw API response saved to {debug_response_file}")

                try:
                    # Extract JSON from content, removing markdown fences if present
                    json_content = content
                    if json_content.startswith("```json"):
                        json_content = json_content.split("```json\n")[1].rsplit("```", 1)[0].strip()

                    extracted_data = json.loads(json_content)
                    # Normalize single object to array
                    if isinstance(extracted_data, dict):
                        logging.debug("Normalizing single JSON object to array")
                        extracted_data = [extracted_data]

                    if not isinstance(extracted_data, list):
                        logging.error(f"API response is not a JSON array: {type(extracted_data)}")
                        continue

                    # Validate extracted data
                    expected_fields = [
                        'اسم الملف', 'رقم الفاتورة', 'التاريخ', 'رقم التعريفي الجبائي', 'بذمة', 'الكمية',
                        'البيانات', 'ثمن القطعة', 'القيمة', 'المجموع', 'اوقفت هذه الفاتورة بقيمة قدره',
                        'رقم الهاتف', 'اسم الشركة'
                    ]
                    for i, item in enumerate(extracted_data):
                        if not isinstance(item, dict):
                            logging.error(f"Item {i} in extracted data is not a dictionary: {type(item)}")
                            continue
                        missing_fields = [f for f in expected_fields if f not in item]
                        if missing_fields:
                            logging.warning(f"Item {i} missing fields: {missing_fields}")
                        logging.debug(f"Extracted fields for invoice {i}: {json.dumps(item, ensure_ascii=False)}")

                    # Save results to Excel
                    if extracted_data:
                        if output_excel is None:
                            output_excel = OUTPUT_EXCEL
                        os.makedirs(os.path.dirname(output_excel), exist_ok=True)
                        df = pd.DataFrame(extracted_data, columns=expected_fields)
                        if os.path.exists(output_excel):
                            existing_df = pd.read_excel(output_excel, engine='openpyxl')
                            df = pd.concat([existing_df, df], ignore_index=True)
                        df.to_excel(output_excel, index=False, engine='openpyxl')
                        # Log Excel contents for verification
                        excel_df = pd.read_excel(output_excel, engine='openpyxl')
                        logging.info(
                            f"Excel file {output_excel} saved with {len(excel_df)} rows. Columns: {list(excel_df.columns)}")
                        logging.debug(f"Excel contents sample: {excel_df.head().to_dict()}")
                        return output_excel
                    else:
                        logging.error("No invoices were processed successfully")
                        return None
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error: {str(e)}", exc_info=True)
                    logging.error(f"Raw content: {content}")
                    continue
            except OpenAIError as e:
                logging.error(f"OpenAI API error (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}", exc_info=True)
                if attempt < MAX_RETRIES - 1:
                    logging.debug(f"Retrying after {RETRY_DELAY} seconds")
                    time.sleep(RETRY_DELAY)
                continue
        logging.error(f"Failed to process invoices after {MAX_RETRIES} attempts")
        return None
    except Exception as e:
        logging.error(f"Error in process_all_invoices: {str(e)}", exc_info=True)
        return None
