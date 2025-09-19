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

# Configure logging with detailed output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('contract_processor_debug.log'),
        logging.StreamHandler()
    ]
)

# Directory and output file
INPUT_DIRECTORY = "contracts"
OUTPUT_DIR = "output"
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "contracts.xlsx")
DEBUG_TEXT_DIR = "debug_text"
DEBUG_IMAGE_DIR = "debug_images"
DEBUG_BASE64_DIR = "debug_base64"

# Open AI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "fake-key")
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
    """Extract text from a PIL image using Open AI gpt-4.1-mini."""
    try:
        base64_image = image_to_base64(image, file_name, image_index)
        if not base64_image:
            logging.error(f"No base64 image generated for {file_name} (image {image_index})")
            return ""

        # Prompt optimized for Arabic contract documents
        prompt = (
            "You are an expert in recognizing text from documents in the MENA region. "
            "Extract all text from the provided image, which is a contract document in Arabic. "
            "The text may be noisy or difficult to read due to document quality. "
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


def process_all_contracts(contract_texts):
    """Send extracted texts to GPT-4 via Open AI API and save results to Excel."""
    try:
        text_sections = []
        for file_name, text in contract_texts:
            logging.debug(f"Processing contract text for {file_name}, text length: {len(text)}")
            text_sections.append(f"Contract: {file_name}\n{text}\n")
        combined_text = "\n".join(text_sections)

        if not combined_text.strip():
            logging.error("No text provided for field extraction")
            return None

        logging.debug(f"Combined text length for API call: {len(combined_text)}")

        prompt = (
            "You are an expert in extracting contract details for the MENA region. "
            "Given the following raw texts extracted from multiple contracts, which are all in Arabic, "
            "extract the specified details for each contract. "
            "Return a list of JSON objects, each containing the following fields in Arabic: "
            "'اسم الملف', 'اسم المؤسسة المقرضة', 'اسم الممثل', 'لقب الممثل', 'رقم البطاقة الشخصية للممثل', 'منصب الممثل', "
            "'اسم المقترض', 'رقم البطاقة الشخصية للمقترض', 'مكان إصدار البطاقة', 'محل الإقامة', "
            "'اسم المنوب عنه', 'صفة المنوب عنه', 'مصدر التوكيل', 'مكان توزيع التوكيل', "
            "'اسم الضامن', 'رقم البطاقة الشخصية للضامن', 'محل إقامة الضامن', "
            "'قيمة القرض (بالأرقام)', 'قيمة القرض (بالحروف)', 'معدل العائد السنوي', 'مدة القرض', "
            "'قيمة العائد الإجمالية (بالأرقام)', 'قيمة العائد الإجمالية (بالحروف)', 'نسبة الخصم المقدم', "
            "'نسبة الأقساط الشهرية', 'مدة السداد', 'رسوم الخدمة الإضافية (بالأرقام)', 'رسوم الخدمة الإضافية (بالحروف)', "
            "'نوع الغرض', 'رقم القرار المرجعي', 'نوع الضمان', 'درجة الرهن', 'قيمة الرهن الإجمالية (بالأرقام)', "
            "'قيمة الرهن الإجمالية (بالحروف)', 'اسم', 'اسم الأب', 'اسم الجد', 'اللقب', 'اسم الأم كامل', 'اسم الزوجة كامل', "
            "'رقم القرض', 'رقم الحساب'. "
            "For fields like 'رقم القرض', 'رقم الحساب', 'رقم البطاقة الشخصية للممثل', 'رقم البطاقة الشخصية للمقترض', "
            "'رقم البطاقة الشخصية للضامن', 'قيمة القرض (بالأرقام)', 'معدل العائد السنوي', 'قيمة العائد الإجمالية (بالأرقام)', "
            "'نسبة الخصم المقدم', 'نسبة الأقساط الشهرية', 'رسوم الخدمة الإضافية (بالأرقام)', 'رقم القرار المرجعي', "
            "'قيمة الرهن الإجمالية (بالأرقام)', convert Arabic numerals (٠=0, ١=1, ٢=2, ٣=3, ٤=4, ٥=5, ٦=6, ٧=7, ٨=8, ٩=9, e.g., ١٠=10) to Western numerals (0-9). "
            "Preserve the original value for fields like 'اسم', 'اسم الأب', 'اسم الجد', 'اسم المقترض', 'اسم الضامن', 'اسم الممثل', 'اسم المنوب عنه', "
            "'لقب الممثل', 'اللقب', 'اسم الأم كامل', 'اسم الزوجة كامل', 'قيمة القرض (بالحروف)', 'قيمة العائد الإجمالية (بالحروف)', "
            "'رسوم الخدمة الإضافية (بالحروف)', 'قيمة الرهن الإجمالية (بالحروف)', as they are names or text in words. "
            "If any information is missing or unclear, indicate it as 'غير موجود'. "
            "Include the contract filename as 'اسم الملف' in each JSON object. "
            "Wrap the JSON output in triple backticks with 'json' identifier (```json\n[...]\n```). "
            "Example output:\n"
            "```json\n"
            "[{\"اسم الملف\":\"contract1.pdf\","
            "\"اسم المؤسسة المقرضة\":\"مصرف الادخار والاستثمار العقاري\","
            "\"اسم الممثل\":\"حسين سالم بلقاسم الدرويش\","
            "\"لقب الممثل\":\"الدرويش\","
            "\"رقم البطاقة الشخصية للممثل\":\"123456\","
            "\"منصب الممثل\":\"مدير عام\","
            "\"اسم المقترض\":\"محمد أحمد\","
            "\"رقم البطاقة الشخصية للمقترض\":\"987654\","
            "\"مكان إصدار البطاقة\":\"وزارة الداخلية\","
            "\"محل الإقامة\":\"طرابلس\","
            "\"اسم المنوب عنه\":\"غير موجود\","
            "\"صفة المنوب عنه\":\"غير موجود\","
            "\"مصدر التوكيل\":\"غير موجود\","
            "\"مكان توزيع التوكيل\":\"غير موجود\","
            "\"اسم الضامن\":\"خالد سالم\","
            "\"رقم البطاقة الشخصية للضامن\":\"123456\","
            "\"محل إقامة الضامن\":\"بنغازي\","
            "\"قيمة القرض (بالأرقام)\":\"10000\","
            "\"قيمة القرض (بالحروف)\":\"عشرة آلاف دينار\","
            "\"معدل العائد السنوي\":\"5\","
            "\"مدة القرض\":\"12 شهر\","
            "\"قيمة العائد الإجمالية (بالأرقام)\":\"500\","
            "\"قيمة العائد الإجمالية (بالحروف)\":\"خمسمائة دينار\","
            "\"نسبة الخصم المقدم\":\"0\","
            "\"نسبة الأقساط الشهرية\":\"1000\","
            "\"مدة السداد\":\"12 شهر\","
            "\"رسوم الخدمة الإضافية (بالأرقام)\":\"500\","
            "\"رسوم الخدمة الإضافية (بالحروف)\":\"خمسمائة دينار\","
            "\"نوع الغرض\":\"شراء عقار\","
            "\"رقم القرار المرجعي\":\"REF123\","
            "\"نوع الضمان\":\"رهن عقاري\","
            "\"درجة الرهن\":\"الأولى\","
            "\"قيمة الرهن الإجمالية (بالأرقام)\":\"2000\","
            "\"قيمة الرهن الإجمالية (بالحروف)\":\"ألفان دينار\","
            "\"اسم\":\"محمد أحمد\","
            "\"اسم الأب\":\"أحمد\","
            "\"اسم الجد\":\"محمود\","
            "\"اللقب\":\"الصغير\","
            "\"اسم الأم كامل\":\"فاطمة سالم\","
            "\"اسم الزوجة كامل\":\"مريم علي\","
            "\"رقم القرض\":\"LN123\","
            "\"رقم الحساب\":\"AC456\"}]\n"
            "```\n"
            f"\n\n{combined_text}"
        )

        for attempt in range(MAX_RETRIES):
            try:
                logging.debug(
                    f"Calling OpenAI API (model: gpt-4) for field extraction (attempt {attempt+1}/{MAX_RETRIES})")

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for extracting contract details."},
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
                debug_response_file = os.path.join(DEBUG_TEXT_DIR, f"all_contracts_response_attempt_{attempt+1}.txt")
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
                        'اسم الملف', 'اسم المؤسسة المقرضة', 'اسم الممثل', 'لقب الممثل', 'رقم البطاقة الشخصية للممثل', 'منصب الممثل',
                        'اسم المقترض', 'رقم البطاقة الشخصية للمقترض', 'مكان إصدار البطاقة', 'محل الإقامة',
                        'اسم المنوب عنه', 'صفة المنوب عنه', 'مصدر التوكيل', 'مكان توزيع التوكيل',
                        'اسم الضامن', 'رقم البطاقة الشخصية للضامن', 'محل إقامة الضامن',
                        'قيمة القرض (بالأرقام)', 'قيمة القرض (بالحروف)', 'معدل العائد السنوي', 'مدة القرض',
                        'قيمة العائد الإجمالية (بالأرقام)', 'قيمة العائد الإجمالية (بالحروف)', 'نسبة الخصم المقدم',
                        'نسبة الأقساط الشهرية', 'مدة السداد', 'رسوم الخدمة الإضافية (بالأرقام)', 'رسوم الخدمة الإضافية (بالحروف)',
                        'نوع الغرض', 'رقم القرار المرجعي', 'نوع الضمان', 'درجة الرهن', 'قيمة الرهن الإجمالية (بالأرقام)',
                        'قيمة الرهن الإجمالية (بالحروف)', 'اسم', 'اسم الأب', 'اسم الجد', 'اللقب', 'اسم الأم كامل', 'اسم الزوجة كامل',
                        'رقم القرض', 'رقم الحساب'
                    ]
                    for i, item in enumerate(extracted_data):
                        if not isinstance(item, dict):
                            logging.error(f"Item {i} in extracted data is not a dictionary: {type(item)}")
                            continue
                        missing_fields = [f for f in expected_fields if f not in item]
                        if missing_fields:
                            logging.warning(f"Item {i} missing fields: {missing_fields}")
                        logging.debug(f"Extracted fields for contract {i}: {json.dumps(item, ensure_ascii=False)}")

                    return extracted_data
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
        logging.error(f"Failed to process contracts after {MAX_RETRIES} attempts")
        return None
    except Exception as e:
        logging.error(f"Error in process_all_contracts: {str(e)}", exc_info=True)
        return None


def main():
    if not os.path.exists(INPUT_DIRECTORY):
        logging.error(f"Directory {INPUT_DIRECTORY} does not exist")
        return

    contract_texts = []
    valid_extensions = ['.pdf', '.jpg', '.jpeg', '.png']
    for file_path in Path(INPUT_DIRECTORY).glob("*"):
        if file_path.suffix.lower() in valid_extensions:
            logging.info(f"Extracting text from {file_path.name}...")
            extracted_text = extract_text_from_file(file_path)
            if not extracted_text:
                logging.error(f"Failed to extract text from {file_path.name}")
                continue
            contract_texts.append((file_path.name, extracted_text))

    if not contract_texts:
        logging.error("No contract texts were extracted successfully")
        return

    logging.info("Processing all contracts with a single API call...")
    results = process_all_contracts(contract_texts)
    if not results:
        logging.error("Failed to process contract details")
        return

    if results:
        df = pd.DataFrame(results, columns=[
            'اسم الملف', 'اسم المؤسسة المقرضة', 'اسم الممثل', 'لقب الممثل', 'رقم البطاقة الشخصية للممثل', 'منصب الممثل',
            'اسم المقترض', 'رقم البطاقة الشخصية للمقترض', 'مكان إصدار البطاقة', 'محل الإقامة',
            'اسم المنوب عنه', 'صفة المنوب عنه', 'مصدر التوكيل', 'مكان توزيع التوكيل',
            'اسم الضامن', 'رقم البطاقة الشخصية للضامن', 'محل إقامة الضامن',
            'قيمة القرض (بالأرقام)', 'قيمة القرض (بالحروف)', 'معدل العائد السنوي', 'مدة القرض',
            'قيمة العائد الإجمالية (بالأرقام)', 'قيمة العائد الإجمالية (بالحروف)', 'نسبة الخصم المقدم',
            'نسبة الأقساط الشهرية', 'مدة السداد', 'رسوم الخدمة الإضافية (بالأرقام)', 'رسوم الخدمة الإضافية (بالحروف)',
            'نوع الغرض', 'رقم القرار المرجعي', 'نوع الضمان', 'درجة الرهن', 'قيمة الرهن الإجمالية (بالأرقام)',
            'قيمة الرهن الإجمالية (بالحروف)', 'اسم', 'اسم الأب', 'اسم الجد', 'اللقب', 'اسم الأم كامل', 'اسم الزوجة كامل',
            'رقم القرض', 'رقم الحساب'
        ])
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if os.path.exists(OUTPUT_EXCEL):
            existing_df = pd.read_excel(OUTPUT_EXCEL, engine='openpyxl')
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_excel(OUTPUT_EXCEL, index=False, engine='openpyxl')
        # Log Excel contents for verification
        excel_df = pd.read_excel(OUTPUT_EXCEL, engine='openpyxl')
        logging.info(f"Excel file {OUTPUT_EXCEL} saved with {len(excel_df)} rows. Columns: {list(excel_df.columns)}")
        logging.debug(f"Excel contents sample: {excel_df.head().to_dict()}")
    else:
        logging.error("No contracts were processed successfully")


if __name__ == "__main__":
    main()
