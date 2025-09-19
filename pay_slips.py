import pdf2image
from PIL import Image, ImageEnhance
import json
import os
import pandas as pd
from pathlib import Path
import time
import logging
import re
import base64
from io import BytesIO
from unicodedata import normalize
from openai import OpenAI, OpenAIError
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('pay_slips_debug.log'),
        logging.StreamHandler()
    ]
)

# Directory for PDFs
PDF_DIRECTORY = "pay_slips"
DEBUG_TEXT_DIR = "debug_text"
OUTPUT_DIR = "output"

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY environment variable not set")
    raise ValueError("OPENAI_API_KEY environment variable is required")
client = OpenAI(api_key=OPENAI_API_KEY)
MAX_RETRIES = 5
RETRY_DELAY = 10
API_TIMEOUT = 180
BATCH_SIZE = 3
MAX_TEXT_LENGTH = 15000
MAX_TOKENS_STRUCTURED = 6000

# API pricing for gpt-4.1-mini (in USD per 1M tokens)
GPT_4_1_MINI_PRICING = {
    'input': 0.15,
    'cached_input': 0.075,
    'output': 0.60
}

def normalize_name(name):
    """
    Fixed name normalization that handles edge cases better.
    """
    if not name or pd.isna(name):
        return 'غير موجود'
    
    name = str(name)
    name = normalize('NFC', name)
    name = re.sub(r'\s+', ' ', name.strip()).lower().replace('_', ' ')
    
    # Remove trailing numbers, dates, and extra suffixes
    name = re.sub(r'(\b\w+\s+\w+)(?:_\d+|\s+\d+|\s*\d{4}(?:\d{2}\d{2})?|\s*removed|\s*rotated)*$', r'\1', name)
    
    parts = name.split()
    if len(parts) >= 2:
        return ' '.join(parts[:2])
    elif len(parts) == 1:
        return parts[0]
    else:
        return 'غير موجود'

def parse_date(date_str):
    if not date_str or pd.isna(date_str) or str(date_str).strip() == '':
        return 'غير متوفر', 'غير متوفر'
    
    date_str = str(date_str).strip().lower()
    month_mappings = {
        # French
        'janvier': 'جانفي', 'février': 'فيفري', 'mars': 'مارس', 'avril': 'أفريل',
        'mai': 'ماي', 'juin': 'جوان', 'juillet': 'جويلية', 'août': 'أوت',
        'septembre': 'سبتمبر', 'octobre': 'أكتوبر', 'novembre': 'نوفمبر', 'décembre': 'ديسمبر',
        # English
        'january': 'جانفي', 'february': 'فيفري', 'march': 'مارس', 'april': 'أفريل',
        'may': 'ماي', 'june': 'جوان', 'july': 'جويلية', 'august': 'أوت',
        'september': 'سبتمبر', 'october': 'أكتوبر', 'november': 'نوفمبر', 'december': 'ديسمبر',
        # Arabic
        'جانفي': 'جانفي', 'فيفري': 'فيفري', 'مارس': 'مارس', 'أفريل': 'أفريل',
        'ماي': 'ماي', 'جوان': 'جوان', 'جويلية': 'جويلية', 'أوت': 'أوت',
        'سبتمبر': 'سبتمبر', 'أكتوبر': 'أكتوبر', 'نوفمبر': 'نوفمبر', 'ديسمبر': 'ديسمبر'
    }
    
    # Try standard month-year format
    parts = date_str.split()
    if len(parts) >= 2:
        month, year = parts[0], parts[-1]
        month = month_mappings.get(month, 'غير متوفر')
        if re.match(r'^\d{4}$', year):
            return month, year
    
    # Try DD/MM/YYYY or YYYY-MM-DD formats
    date_formats = [r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})']
    for fmt in date_formats:
        match = re.match(fmt, date_str)
        if match:
            groups = match.groups()
            if fmt.startswith(r'(\d{4})'):
                year, month = groups[0], groups[1]
            else:
                month, year = groups[1], groups[2]
            month_num = int(month) if month.isdigit() else 'غير متوفر'
            month = list(month_mappings.values())[month_num - 1] if 1 <= month_num <= 12 else 'غير متوفر'
            return month, year
    
    return 'غير متوفر', 'غير متوفر'

def chunk_text(text, max_length=MAX_TEXT_LENGTH):
    """Split text into chunks if it's too long."""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    pages = text.split("--- Page ")
    current_chunk = ""
    
    for page in pages:
        if not page.strip():
            continue
            
        page_text = "--- Page " + page if page != pages[0] else page
        
        if len(current_chunk) + len(page_text) > max_length and current_chunk:
            chunks.append(current_chunk)
            current_chunk = page_text
        else:
            current_chunk += page_text if not current_chunk else "\n" + page_text
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def extract_structured_data_with_retry(client, prompt, pdf_path, max_retries=MAX_RETRIES):
    """Execute API call with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            logging.debug(f"API call attempt {attempt+1}/{max_retries} for {pdf_path}")
            
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for pay slip data extraction."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=MAX_TOKENS_STRUCTURED,
                timeout=API_TIMEOUT,
                stream=False
            )
            
            return response, None
            
        except requests.exceptions.Timeout as e:
            logging.warning(f"Timeout error for {pdf_path} (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                logging.info(f"Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                return None, f"Timeout after {max_retries} attempts"
                
        except OpenAIError as e:
            logging.error(f"OpenAI API error for {pdf_path} (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                logging.info(f"Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                return None, f"OpenAI API error after {max_retries} attempts: {str(e)}"
                
        except Exception as e:
            logging.error(f"Unexpected error for {pdf_path} (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                delay = RETRY_DELAY
                logging.info(f"Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                return None, f"Unexpected error after {max_retries} attempts: {str(e)}"
    
    return None, "Max retries exceeded"

def validate_extracted_data(data):
    """Validate if extracted data contains critical fields with valid values."""
    critical_fields = ['اسم الشركة', 'التاريخ', 'رقم بطاقة التعريف']
    missing_fields = []
    
    for item in data:
        item_missing = []
        for field in critical_fields:
            if field not in item or item[field] == 'غير موجود':
                item_missing.append(field)
        
        if not item_missing:
            return True
        missing_fields.append((item, item_missing))
    
    if all('رقم بطاقة التعريف' in missing and len(missing) == 1 for _, missing in missing_fields):
        logging.info("رقم بطاقة التعريف missing in all items, using placeholder 'غير متوفر'")
        for item in data:
            if item['رقم بطاقة التعريف'] == 'غير موجود':
                item['رقم بطاقة التعريف'] = 'غير متوفر'
        return True
    
    logging.warning(f"Validation failed due to missing or invalid fields: {missing_fields}")
    return False

def image_to_base64(image, file_name="unknown", image_index=0):
    """Convert a PIL Image to base64 string after resizing and enhancing."""
    try:
        logging.debug(f"Converting image to base64 for {file_name} (page {image_index+1})")
        max_size = (800, 800)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        image = ImageEnhance.Contrast(image).enhance(1.5)
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True)
        base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        os.makedirs(DEBUG_TEXT_DIR, exist_ok=True)
        debug_base64_file = os.path.join(DEBUG_TEXT_DIR, f"{Path(file_name).stem}_page_{image_index+1}_base64.txt")
        with open(debug_base64_file, 'w', encoding='utf-8') as f:
            f.write(base64_string)
        logging.debug(f"Saved base64 string to {debug_base64_file}, length: {len(base64_string)}")
        return base64_string
    except Exception as e:
        logging.error(f"Error converting image to base64 for {file_name} (page {image_index+1}): {str(e)}")
        return None

def extract_text_from_image_batch(images, file_name, start_index):
    """Extract raw text from a batch of images using gpt-4.1-mini."""
    try:
        content = [{
            "type": "text",
            "text": (
                f"Extract text from {len(images)} pay slip pages. For each page, provide the text prefixed with 'Page {{page_number}}:' using (Page {start_index+1}, Page {start_index+2}, ...)."
                "where i is the page index in the batch (0-based). Handle Arabic, French, and English text, correcting common OCR errors "
                "(e.g., 'SalairedeBase' to 'Salaire de Base', merged words, misread characters). Return the text for each page clearly separated."
            )
        }]
        valid_images = []
        for i, image in enumerate(images):
            base64_image = image_to_base64(image, file_name, start_index + i)
            if base64_image:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "auto"}})
                valid_images.append(i)
            else:
                logging.warning(f"Failed to generate base64 for {file_name} (page {start_index+i+1})")
        if not valid_images:
            logging.error(f"No valid images for {file_name} (pages {start_index+1} to {start_index+len(images)})")
            return [""] * len(images), 0.0
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0,
                    max_tokens=3000,
                    timeout=API_TIMEOUT
                )
                text = response.choices[0].message.content
                if not isinstance(text, str):
                    logging.error(f"API response content is not a string for {file_name} (pages {start_index+1} to {start_index+len(images)}): {type(text)}")
                    return [""] * len(images), 0.0
                text = text.strip()
                os.makedirs(DEBUG_TEXT_DIR, exist_ok=True)
                debug_response_file = os.path.join(DEBUG_TEXT_DIR, f"{Path(file_name).stem}_pages_{start_index+1}_to_{start_index+len(images)}_raw_response.txt")
                with open(debug_response_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                logging.debug(f"Saved raw API response to {debug_response_file}")
                texts = [t.strip() for t in text.split("Page ")[1:]] if "Page " in text else [text] * len(valid_images)
                texts = texts + [""] * (len(valid_images) - len(texts))
                result_texts = [""] * len(images)
                for i, t in zip(valid_images, texts):
                    result_texts[i] = t
                    debug_text_file = os.path.join(DEBUG_TEXT_DIR, f"{Path(file_name).stem}_page_{start_index+i+1}_raw.txt")
                    with open(debug_text_file, 'w', encoding='utf-8') as f:
                        f.write(t)
                    logging.debug(f"Saved raw extracted text to {debug_text_file}")
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                input_cost = (input_tokens / 1_000_000) * GPT_4_1_MINI_PRICING['input']
                output_cost = (output_tokens / 1_000_000) * GPT_4_1_MINI_PRICING['output']
                total_cost = input_cost + output_cost
                logging.info(f"gpt-4.1-mini API Cost for {file_name} (pages {start_index+1} to {start_index+len(images)}):")
                logging.info(f"  Input Tokens: {input_tokens} (${input_cost:.6f})")
                logging.info(f"  Output Tokens: {output_tokens} (${output_cost:.6f})")
                logging.info(f"  Total Cost: ${total_cost:.6f}")
                return result_texts, total_cost
            except (OpenAIError, requests.exceptions.Timeout) as e:
                logging.error(f"OpenAI API error for {file_name} (pages {start_index+1} to {start_index+len(images)}, attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    logging.info(f"Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue
        logging.error(f"Failed to extract text from {file_name} (pages {start_index+1} to {start_index+len(images)}) after {MAX_RETRIES} attempts")
        return [""] * len(images), 0.0
    except Exception as e:
        logging.error(f"Error extracting text from {file_name} (pages {start_index+1} to {start_index+len(images)}): {str(e)}")
        return [""] * len(images), 0.0

def extract_structured_data(texts, client_name, pdf_path):
    """Extract structured pay slip data from combined text using gpt-4.1-mini."""
    try:
        combined_text = "\n".join(f"--- Page {i+1} ---\n{text}" for i, text in enumerate(texts))
        
        if not combined_text.strip():
            logging.error(f"No text to process for {pdf_path}")
            return [], 0.0
        
        logging.info(f"Processing {len(combined_text)} characters for {pdf_path}")
        
        text_chunks = chunk_text(combined_text)
        logging.info(f"Split text into {len(text_chunks)} chunks for {pdf_path}")
        
        all_extracted_data = []
        total_cost = 0.0
        
        for chunk_idx, text_chunk in enumerate(text_chunks):
            logging.info(f"Processing chunk {chunk_idx + 1}/{len(text_chunks)} for {pdf_path}")
            
            prompt = (
                "Extract pay slip data from the following text. Return a JSON array with one JSON object per page, using Arabic field names.\n"
                "Fixed fields: 'اسم الملف', 'الاسم', 'رقم الصفحة', 'اسم الشركة', 'نوع الوثيقة', 'التاريخ', "
                "'رقم CNSS صاحب العمل', 'رقم بطاقة التعريف', 'الاسم الكامل', 'صافي الدفع'.\n"
                "Dynamic fields: For each 'الوصف' (description/libelle) item in the table (under 'المستحقات' or 'الاقتطاعات'), "
                "include its value as a separate field in the JSON object, using the Arabic translation of the 'الوصف' value as the field name. "
                "If the 'الوصف' value is in French or English, translate it to Arabic using the following mappings or best judgment for unmapped terms:\n"
                "- 'net à payer' or 'Net à Payer' or 'NET A PAYER' → 'صافي الدفع'\n"
                "- 'Salaire de base' or 'SALAIRE DE BASE' (from المستحقات) → 'الاجر الاساسي salaire de base'\n"
                "- 'SALAIRE BRUT' (from المستحقات) → 'الاجر الإجمالي salaire brut'\n"
                "- 'Nombre de jours fériés' (from المستحقات) → 'عدد الأيام العطل'\n"
                "- 'Prime de présence' (from المستحقات) → 'بريم الحضور'\n"
                "- 'Prime de transport' (from المستحقات) → 'بريم النقل'\n"
                "- 'Prime de rendement' (from المستحقات) → 'بريم الإنتاجية'\n"
                "- 'Prime de douche' (from المستحقات) → 'بريم الدوش'\n"
                "- 'Jours fériés' (from المستحقات) → 'أيام العطل'\n"
                "- 'C.N.S.S.' (from الاقتطاعات) → 'C.N.S.S.'\n"
                "- 'Retenues impôt (I.R.P.P)' or 'IRPP' (from الاقتطاعات) → 'ضريبة الدخل IRPP'\n"
                "- 'Avance sur salaire' (from الاقتطاعات) → 'السلفة على الأجر'\n"
                "- 'N° C.I.N' or 'CIN' or 'National ID' → 'رقم بطاقة التعريف'\n"
                "For any other 'الوصف' values, provide a reasonable Arabic translation based on context (e.g., 'Prime de panier' → 'بريم السلة'). "
                "Include the value from 'المستحقات' or 'الاقتطاعات' for each 'الوصف' item. "
                "Use 'غير متوفر' for missing values. Ensure ALL fixed fields are included in every JSON object. "
                "If no 'الوصف' items are found on a page, include only the fixed fields.\n"
                f"Filename: {Path(pdf_path).name}, Client: {client_name}\n"
                f"Text:\n{text_chunk}"
            )
            
            response, error = extract_structured_data_with_retry(client, prompt, pdf_path)
            
            if error:
                logging.error(f"Failed to process chunk {chunk_idx + 1} for {pdf_path}: {error}")
                continue
                
            if not response:
                logging.error(f"No response for chunk {chunk_idx + 1} for {pdf_path}")
                continue
            
            content = response.choices[0].message.content
            if not isinstance(content, str):
                logging.error(f"API response content is not a string for {pdf_path} chunk {chunk_idx + 1}: {type(content)}")
                continue
            
            content = content.strip()
            
            os.makedirs(DEBUG_TEXT_DIR, exist_ok=True)
            debug_raw_response_file = os.path.join(DEBUG_TEXT_DIR, f"{Path(pdf_path).stem}_chunk_{chunk_idx + 1}_raw_response.txt")
            with open(debug_raw_response_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.debug(f"Saved raw API response to {debug_raw_response_file}")
            
            content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
            content = content.strip('`')
            content = re.sub(r'^[^[]*(\[.*\])', r'\1', content, flags=re.MULTILINE)
            if not content.endswith(']'):
                content += ']'
            
            try:
                extracted_data = json.loads(content)
                if not isinstance(extracted_data, list):
                    logging.error(f"API response for {pdf_path} chunk {chunk_idx + 1} is not a JSON array")
                    continue
                
                for item in extracted_data:
                    item['الاسم'] = client_name
                
                all_extracted_data.extend(extracted_data)
                
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                input_cost = (input_tokens / 1_000_000) * GPT_4_1_MINI_PRICING['input']
                output_cost = (output_tokens / 1_000_000) * GPT_4_1_MINI_PRICING['output']
                chunk_cost = input_cost + output_cost
                total_cost += chunk_cost
                
                logging.info(f"gpt-4.1-mini API Cost for chunk {chunk_idx + 1} of {pdf_path}:")
                logging.info(f"  Input Tokens: {input_tokens} (${input_cost:.6f})")
                logging.info(f"  Output Tokens: {output_tokens} (${output_cost:.6f})")
                logging.info(f"  Chunk Cost: ${chunk_cost:.6f}")
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error for {pdf_path} chunk {chunk_idx + 1}: {e}")
                logging.error(f"Cleaned content: {content}")
                try:
                    last_valid_index = content.rfind('},')
                    if last_valid_index != -1:
                        partial_content = content[:last_valid_index + 1] + ']'
                        extracted_data = json.loads(partial_content)
                        if isinstance(extracted_data, list):
                            for item in extracted_data:
                                item['الاسم'] = client_name
                            all_extracted_data.extend(extracted_data)
                            logging.info(f"Recovered {len(extracted_data)} items from partial JSON for {pdf_path} chunk {chunk_idx + 1}")
                        else:
                            logging.error(f"Partial JSON is not a list for {pdf_path} chunk {chunk_idx + 1}")
                    else:
                        logging.error(f"Could not recover partial JSON for {pdf_path} chunk {chunk_idx + 1}")
                except json.JSONDecodeError as partial_e:
                    logging.error(f"Partial JSON decode failed for {pdf_path} chunk {chunk_idx + 1}: {partial_e}")
                continue
        
        if all_extracted_data:
            if validate_extracted_data(all_extracted_data):
                logging.info(f"Data extracted for {pdf_path} with {len(all_extracted_data)} items")
                logging.info(f"Total structured extraction cost for {pdf_path}: ${total_cost:.6f}")
                return all_extracted_data, total_cost
            else:
                logging.warning(f"Validation failed for {pdf_path}")
                return [], total_cost
        else:
            logging.error(f"No valid data extracted from {pdf_path}")
            return [], total_cost
            
    except Exception as e:
        logging.error(f"Error processing structured data for {pdf_path}: {str(e)}")
        return [], 0.0

def extract_text_from_pdf(pdf_path):
    """Extract structured pay slip data from PDF using gpt-4.1-mini."""
    try:
        logging.info(f"Converting PDF {pdf_path} to images")
        images = pdf2image.convert_from_path(pdf_path, dpi=100, thread_count=2)
        if not images:
            logging.error(f"No pages found in {pdf_path}")
            return [], 0.0
        if len(images) > 10:
            logging.warning(f"PDF {pdf_path} has {len(images)} pages, truncating to 10")
            images = images[:10]
        client_name = normalize_name(Path(pdf_path).stem)
        page_texts = []
        total_text_cost = 0.0
        for i in range(0, len(images), BATCH_SIZE):
            batch_images = images[i:i + BATCH_SIZE]
            texts, batch_cost = extract_text_from_image_batch(batch_images, pdf_path, i)
            total_text_cost += batch_cost
            page_texts.extend(texts)
        if not any(page_texts):
            logging.error(f"No text extracted from {pdf_path}")
            return [], total_text_cost
        results, struct_cost = extract_structured_data(page_texts, client_name, pdf_path)
        total_cost = total_text_cost + struct_cost
        logging.info(f"Total API Cost for {pdf_path}: ${total_cost:.6f} (Text Extraction: ${total_text_cost:.6f}, Structured Extraction: ${struct_cost:.6f})")
        return results, total_cost
    except pdf2image.exceptions.PDFPageCountError:
        logging.error(f"Invalid PDF structure in {pdf_path}")
        return [], 0.0
    except Exception as e:
        logging.error(f"Error extracting data from {pdf_path}: {e}")
        return [], 0.0

def clean_column_name(col):
    """
    Clean column names by removing suffixes like .1, .2, etc., and normalizing variations.
    """
    col = str(col).strip()
    # Remove numeric suffixes (.1, .2, etc.)
    col = re.sub(r'\.\d+$', '', col)
    # Normalize spaces and remove extra characters
    col = normalize('NFC', col)
    col = re.sub(r'\s+', ' ', col).strip()
    return col

def save_by_client(all_results, output_dir):
    """
    Save pay slip data to Excel files by client, ensuring no duplicate columns and a specific column order.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Fixed columns in desired order
    fixed_columns = [
        'تاريخ الوثيقة',
        'الشهر',
        'السنة',
        'نوع الوثيقة',
        'اسم المؤجر',
        'اسم الاجير',
        'عدد الصندوق الوطني للضمان الاجتماعي',
        'عدد بطاقة التعريف',
        'صافي الدفع'
    ]
    
    # Comprehensive column mapping for cleaning duplicates
    column_mapping = {
        'التاريخ': 'تاريخ الوثيقة',
        'تاريخ الوثيقة (شهر ديسمبر)': 'تاريخ الوثيقة',
        'اسم الشركة': 'اسم المؤجر',
        'الاسم الكامل': 'اسم الاجير',
        'رقم CNSS صاحب العمل': 'عدد الصندوق الوطني للضمان الاجتماعي',
        'رقم بطاقة التعريف': 'عدد بطاقة التعريف',
        'N° C.I.N': 'عدد بطاقة التعريف',
        'CIN': 'عدد بطاقة التعريف',
        'National ID': 'عدد بطاقة التعريف',
        'رقم التسجيل': 'عدد بطاقة التعريف',
        'صافي الدفع': 'صافي الدفع',
        'net à payer': 'صافي الدفع',
        'Net à Payer': 'صافي الدفع',
        'NET A PAYER': 'صافي الدفع',
        'SALAIRE DE BASE': 'الاجر الاساسي salaire de base',
        'Salaire de base': 'الاجر الاساسي salaire de base',
        'SALAIRE BRUT': 'الاجر الإجمالي salaire brut',
        'Nombre de jours fériés': 'عدد الأيام العطل',
        'Prime de présence': 'بريم الحضور',
        'Prime de transport': 'بريم النقل',
        'Prime de rendement': 'بريم الإنتاجية',
        'Prime de douche': 'بريم الدوش',
        'Jours fériés': 'أيام العطل',
        'C.N.S.S.': 'C.N.S.S.',
        'Retenues impôt (I.R.P.P)': 'ضريبة الدخل IRPP',
        'IRPP': 'ضريبة الدخل IRPP',
        'Avance sur salaire': 'السلفة على الأجر',
        'نوع الوثيقة (بطاقة خلاص)': 'نوع الوثيقة'
    }
    
    # Collect all unique 'الوصف' values to create dynamic columns
    dynamic_columns = set()
    for result in all_results:
        for key in result.keys():
            cleaned_key = clean_column_name(key)
            mapped_key = column_mapping.get(cleaned_key, cleaned_key)
            if mapped_key not in fixed_columns and mapped_key not in ['اسم الملف', 'الاسم', 'رقم الصفحة']:
                dynamic_columns.add(mapped_key)
    dynamic_columns = sorted(list(dynamic_columns))  # Sort for consistent column order
    
    # Combine fixed and dynamic columns in the desired order
    final_columns = fixed_columns + dynamic_columns
    
    client_files = {}
    existing_files = list(Path(output_dir).glob('payslip_*.xlsx'))
    
    for result in all_results:
        client_name = result.get('الاسم', 'غير موجود')
        if client_name == 'غير موجود':
            logging.warning(f"Skipping result with missing client name")
            continue
        
        # Normalize client name
        normalized_client_name = normalize_name(client_name)
        output_excel = None
        
        # Find existing file
        for existing_file in existing_files:
            try:
                existing_client_name = normalize_name(
                    existing_file.stem.replace('payslip_', '')
                )
                if normalized_client_name == existing_client_name:
                    output_excel = str(existing_file)
                    break
            except Exception as e:
                logging.warning(f"Error checking existing file {existing_file}: {e}")
                continue
        
        # Create new file if no match found
        if not output_excel:
            safe_client_name = normalized_client_name.replace(' ', '_')
            output_excel = os.path.join(output_dir, f"payslip_{safe_client_name}.xlsx")
        
        # Parse date
        date_str = result.get('تاريخ الوثيقة', result.get('التاريخ', 'غير متوفر'))
        month, year = parse_date(date_str)
        
        # Create mapped result with cleaned column names
        mapped_result = {}
        for key in result.keys():
            cleaned_key = clean_column_name(key)
            mapped_key = column_mapping.get(cleaned_key, cleaned_key)
            if mapped_key in final_columns or mapped_key in ['اسم الملف', 'الاسم', 'رقم الصفحة']:
                mapped_result[mapped_key] = result[key]
        
        # Ensure all fixed columns are present
        mapped_result.update({
            'تاريخ الوثيقة': date_str,
            'الشهر': month,
            'السنة': year,
            'نوع الوثيقة': result.get('نوع الوثيقة', 'غير متوفر'),
            'اسم المؤجر': result.get('اسم الشركة', 'غير متوفر'),
            'اسم الاجير': result.get('الاسم الكامل', 'غير متوفر'),
            'عدد الصندوق الوطني للضمان الاجتماعي': result.get('رقم CNSS صاحب العمل', 'غير متوفر'),
            'عدد بطاقة التعريف': result.get('رقم بطاقة التعريف', 'غير متوفر'),
            'صافي الدفع': result.get('صافي الدفع', 'غير متوفر')
        })
        
        # Add dynamic columns
        for col in dynamic_columns:
            if col not in mapped_result:
                mapped_result[col] = 'غير متوفر'
        
        if output_excel not in client_files:
            client_files[output_excel] = []
        client_files[output_excel].append(mapped_result)
    
    # Save files
    saved_files = []
    for output_excel, results in client_files.items():
        try:
            # Create new DataFrame
            new_df = pd.DataFrame(results)
            
            # Clean and map column names to prevent duplicates
            new_df.columns = [clean_column_name(col) for col in new_df.columns]
            new_df = new_df.rename(columns=column_mapping)
            
            # Check for duplicate columns and merge them
            cols = new_df.columns
            duplicates = [col for col in set(cols) if cols.tolist().count(col) > 1]
            if duplicates:
                logging.warning(f"Duplicate columns found in new data for {output_excel}: {duplicates}")
                for dup_col in duplicates:
                    dup_cols = [c for c in new_df.columns if c == dup_col]
                    combined = new_df[dup_cols].bfill(axis=1).iloc[:, 0].where(
                        new_df[dup_cols].ne('غير متوفر').any(axis=1), new_df[dup_cols].iloc[:, 0]
                    )
                    new_df = new_df.drop(columns=dup_cols)
                    new_df[dup_col] = combined
                logging.info(f"Merged duplicate columns: {duplicates}")
            
            # Add missing fixed columns
            for col in fixed_columns:
                if col not in new_df.columns:
                    if col in ['الشهر', 'السنة'] and 'تاريخ الوثيقة' in new_df.columns:
                        new_df['الشهر'] = new_df['تاريخ الوثيقة'].apply(lambda x: parse_date(x)[0])
                        new_df['السنة'] = new_df['تاريخ الوثيقة'].apply(lambda x: parse_date(x)[1])
                    else:
                        new_df[col] = 'غير متوفر'
            
            # Ensure only final_columns are included and in correct order
            new_df = new_df[[col for col in final_columns if col in new_df.columns]]
            
            # Handle existing file
            if os.path.exists(output_excel):
                try:
                    existing_df = pd.read_excel(output_excel, engine='openpyxl')
                    
                    # Clean and map existing column names
                    existing_df.columns = [clean_column_name(col) for col in existing_df.columns]
                    existing_df = existing_df.rename(columns=column_mapping)
                    
                    # Check for duplicate columns in existing file
                    cols = existing_df.columns
                    duplicates = [col for col in set(cols) if cols.tolist().count(col) > 1]
                    if duplicates:
                        logging.warning(f"Duplicate columns found in existing file {output_excel}: {duplicates}")
                        for dup_col in duplicates:
                            dup_cols = [c for c in existing_df.columns if c == dup_col]
                            combined = existing_df[dup_cols].bfill(axis=1).iloc[:, 0].where(
                                existing_df[dup_cols].ne('غير متوفر').any(axis=1), existing_df[dup_cols].iloc[:, 0]
                            )
                            existing_df = existing_df.drop(columns=dup_cols)
                            existing_df[dup_col] = combined
                        logging.info(f"Merged duplicate columns in existing file: {duplicates}")
                    
                    # Add missing columns
                    for col in fixed_columns:
                        if col not in existing_df.columns:
                            if col in ['الشهر', 'السنة'] and 'تاريخ الوثيقة' in existing_df.columns:
                                existing_df['الشهر'] = existing_df['تاريخ الوثيقة'].apply(lambda x: parse_date(x)[0])
                                existing_df['السنة'] = existing_df['تاريخ الوثيقة'].apply(lambda x: parse_date(x)[1])
                            else:
                                existing_df[col] = 'غير متوفر'
                    
                    # Ensure column order
                    existing_df = existing_df[[col for col in final_columns if col in existing_df.columns]]
                    
                    # Combine DataFrames, dropping duplicates based on key fields
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(
                        subset=['تاريخ الوثيقة', 'اسم الاجير', 'عدد بطاقة التعريف'], keep='last'
                    )
                    
                except Exception as e:
                    logging.warning(f"Error reading existing file {output_excel}: {e}")
                    combined_df = new_df
            else:
                combined_df = new_df
            
            # Save the file
            combined_df.to_excel(output_excel, index=False, engine='openpyxl')
            saved_files.append(output_excel)
            logging.info(f"Saved {output_excel} with {len(combined_df)} rows")
            
        except Exception as e:
            logging.error(f"Error saving {output_excel}: {e}")
            continue
    
    return saved_files

if __name__ == "__main__":
    if not os.path.exists(PDF_DIRECTORY):
        logging.error(f"Directory {PDF_DIRECTORY} does not exist")
    else:
        all_results = []
        total_cost = 0.0
        pdf_count = 0
        for pdf_file in Path(PDF_DIRECTORY).glob("*.pdf"):
            pdf_count += 1
            logging.info(f"Processing {pdf_file}...")
            start_time = time.time()
            results, pdf_cost = extract_text_from_pdf(str(pdf_file))
            total_cost += pdf_cost
            if isinstance(results, list):
                if results:
                    all_results.extend(results)
                else:
                    logging.warning(f"No results returned for {pdf_file}")
            else:
                logging.error(f"Expected list for results from {pdf_file}, got {type(results)}")
            logging.info(f"API Processing Summary for {pdf_file}:")
            logging.info(f"  Duration: {time.time() - start_time:.2f} seconds")
            logging.info(f"  Cost: ${pdf_cost:.6f}")
        if all_results:
            saved_files = save_by_client(all_results, OUTPUT_DIR)
            logging.info(f"Saved {len(saved_files)} Excel files: {saved_files}")
            logging.info(f"Processed {pdf_count} PDFs with {len(all_results)} records")
            logging.info(f"Total API Cost for all documents: ${total_cost:.6f}")
        else:
            logging.error("No pay slips processed successfully")
