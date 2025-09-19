import re
import time
from flask import Blueprint, render_template, request, send_file, jsonify, current_app
import os
import uuid
import logging
import shutil
from pathlib import Path
import openpyxl
import pandas as pd
from invoice import extract_text_from_pdf as extract_invoice_text, process_all_invoices as process_regular_invoices
from pay_slips import extract_text_from_pdf as extract_payslip_text, save_by_client, validate_extracted_data
from handwritten_invoice import extract_text_from_file as extract_handwritten_text, process_all_invoices as process_handwritten_invoices
from contracts import extract_text_from_file as extract_contract_text, process_all_contracts
from unicodedata import normalize
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

bp = Blueprint('main', __name__)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def normalize_name(name):
    """Normalize a name by removing extra spaces, diacritics, and trailing numbers/dates, keeping name_surname format."""
    name = normalize('NFC', name)
    name = re.sub(r'\s+', ' ', name.strip()).lower().replace('_', ' ')
    name = re.sub(r'(\b\w+\s+\w+)(?:_\d+|\s+\d+|\s*\d{4}(?:\d{2}\d{2})?|\s*\w+)*$', r'\1', name)
    parts = name.split()
    if len(parts) >= 2:
        return ' '.join(parts[:2])
    return ' '.join(parts) if parts else 'غير موجود'

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

def validate_excel_columns(file_path, fixed_columns=None):
    """Validate and update Excel file columns, dynamically handling new columns and duplicates."""
    fixed_columns = fixed_columns or [
        'تاريخ الوثيقة', 'الشهر', 'السنة', 'نوع الوثيقة', 'اسم المؤجر', 'اسم الاجير',
        'عدد الصندوق الوطني للضمان الاجتماعي', 'عدد بطاقة التعريف', 'صافي الدفع'
    ]
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
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        logging.info(f"Columns in {file_path} before validation: {df.columns.tolist()}")
        
        # Clean column names
        df.columns = [clean_column_name(col) for col in df.columns]
        df = df.rename(columns=column_mapping)
        
        # Check for duplicate columns
        cols = df.columns
        duplicates = [col for col in set(cols) if cols.tolist().count(col) > 1]
        if duplicates:
            logging.warning(f"Duplicate columns found in {file_path}: {duplicates}")
            for dup_col in duplicates:
                # Select columns with the duplicate name
                dup_cols = [c for c in df.columns if c == dup_col]
                if len(dup_cols) > 1:
                    # Combine duplicate columns, prioritizing non-'غير متوفر' values
                    combined = df[dup_cols].bfill(axis=1).iloc[:, 0].where(
                        df[dup_cols].ne('غير متوفر').any(axis=1), df[dup_cols].iloc[:, 0]
                    )
                    # Drop all duplicate columns
                    df = df.drop(columns=dup_cols)
                    # Add back the merged column
                    df[dup_col] = combined
            logging.info(f"Merged duplicate columns: {duplicates}")
        
        # Get all columns, including dynamic ones
        existing_columns = list(df.columns)
        dynamic_columns = sorted([col for col in existing_columns if col not in fixed_columns])
        all_columns = fixed_columns + dynamic_columns
        
        # Add missing fixed columns
        added_columns = []
        for col in fixed_columns:
            if col not in df.columns:
                if col in ['الشهر', 'السنة'] and 'تاريخ الوثيقة' in df.columns:
                    df['الشهر'] = df['تاريخ الوثيقة'].apply(lambda x: parse_date(x)[0])
                    df['السنة'] = df['تاريخ الوثيقة'].apply(lambda x: parse_date(x)[1])
                    added_columns.extend(['الشهر', 'السنة'])
                else:
                    df[col] = 'غير متوفر'
                    added_columns.append(col)
        
        # Ensure column order
        df = df[[col for col in all_columns if col in df.columns]]
        df.to_excel(file_path, index=False, engine='openpyxl')
        logging.info(f"Updated {file_path} with columns: {df.columns.tolist()}")
        if added_columns:
            logging.info(f"Added missing columns: {added_columns}")
        return True
    except Exception as e:
        logging.error(f"Error validating {file_path}: {e}")
        return False

@bp.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@bp.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and process them based on document type."""
    try:
        if 'files' not in request.files or 'doc_type' not in request.form:
            logging.error("No files or document type provided in request")
            return jsonify({'error': 'No files or document type provided'}), 400
        
        doc_type = request.form['doc_type']
        files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            logging.error("No files selected for upload")
            return jsonify({'error': 'No files selected'}), 400
        
        if doc_type not in ['invoices', 'pay_slips', 'handwritten_invoices', 'contracts']:
            logging.error(f"Invalid document type: {doc_type}")
            return jsonify({'error': 'Invalid document type'}), 400
        
        session_id = str(uuid.uuid4())
        session_upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], session_id, doc_type)
        os.makedirs(session_upload_dir, exist_ok=True)
        
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join(session_upload_dir, filename)
                file.save(file_path)
                saved_files.append(file_path)
            else:
                logging.error(f"Invalid file: {file.filename}")
                return jsonify({'error': f'Invalid file: {file.filename}'}), 400
        
        output_dir = os.path.join('output')
        os.makedirs(output_dir, exist_ok=True)
        
        if doc_type == 'invoices':
            invoice_texts = []
            for pdf_path in saved_files:
                logging.info(f"Extracting text from {pdf_path}...")
                extracted_text = extract_invoice_text(pdf_path)
                if extracted_text:
                    invoice_texts.append((Path(pdf_path).name, extracted_text))
                else:
                    logging.error(f"Failed to extract text from {pdf_path}")
            
            if invoice_texts:
                logging.info("Processing all invoices...")
                results = process_regular_invoices(invoice_texts)
                if results:
                    df = pd.DataFrame(results, columns=[
                        'اسم الملف', 'رقم الفاتورة', 'التاريخ', 'المبلغ الإجمالي', 'اسم المورد', 
                        'عنوان المورد', 'رقم التعريفي الجبائي'
                    ])
                    output_excel = os.path.join(output_dir, "invoices.xlsx")
                    if os.path.exists(output_excel):
                        existing_df = pd.read_excel(output_excel, engine='openpyxl')
                        df = pd.concat([existing_df, df], ignore_index=True)
                    df.to_excel(output_excel, index=False, engine='openpyxl')
                    logging.info(f"Results appended to {output_excel}")
                    if not os.path.exists(output_excel):
                        logging.error(f"Output file {output_excel} was not created")
                        return jsonify({'error': 'Output file not created'}), 500
                    return jsonify({
                        'status': 'success',
                        'download_path': f'/download/output/invoices.xlsx'
                    })
                else:
                    logging.error("Failed to process invoices")
                    return jsonify({'error': 'Failed to process invoices'}), 500
            else:
                logging.error("No invoice texts extracted")
                return jsonify({'error': 'No invoice texts extracted'}), 500
        
        elif doc_type == 'pay_slips':
            all_results = []
            processing_date = time.strftime('%Y-%m-%d %H:%M:%S')
            for pdf_path in saved_files:
                logging.info(f"Processing {pdf_path}...")
                results, cost = extract_payslip_text(pdf_path)
                if isinstance(results, list) and results:
                    if validate_extracted_data(results):
                        for result in results:
                            result['تاريخ المعالجة'] = processing_date
                        all_results.extend(results)
                    else:
                        logging.error(f"Validation failed for {pdf_path}")
                else:
                    logging.error(f"Failed to process {pdf_path}: results is {type(results)} or empty")
            
            if all_results:
                output_files = save_by_client(all_results, output_dir=output_dir)
                if output_files:
                    download_paths = []
                    for file in output_files:
                        if validate_excel_columns(file):
                            download_paths.append(f'/download/output/{os.path.basename(file)}')
                        else:
                            logging.warning(f"Skipping invalid file {file}")
                    if download_paths:
                        logging.info(f"Returning download paths: {download_paths}")
                        return jsonify({
                            'status': 'success',
                            'download_paths': download_paths
                        })
                    else:
                        logging.error("No valid output files generated for pay slips")
                        return jsonify({'error': 'No valid output files generated'}), 500
                else:
                    logging.error("No output files generated for pay slips")
                    return jsonify({'error': 'No output files generated'}), 500
            else:
                logging.error("No pay slips processed successfully")
                return jsonify({'error': 'No pay slips processed successfully'}), 500
        
        elif doc_type == 'handwritten_invoices':
            invoice_texts = []
            for file_path in saved_files:
                logging.info(f"Extracting text from {file_path}...")
                extracted_text = extract_handwritten_text(file_path)
                if extracted_text:
                    invoice_texts.append((Path(file_path).name, extracted_text))
                else:
                    logging.error(f"Failed to extract text from {file_path}")
            
            if invoice_texts:
                logging.info("Processing all handwritten invoices...")
                results = process_handwritten_invoices(invoice_texts)
                if results:
                    output_excel = os.path.join(output_dir, "handwritten_invoices.xlsx")
                    if os.path.exists(output_excel):
                        existing_df = pd.read_excel(output_excel, engine='openpyxl')
                        df = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
                    else:
                        df = pd.DataFrame(results)
                    df.to_excel(output_excel, index=False, engine='openpyxl')
                    logging.info(f"Results appended to {output_excel}")
                    if not os.path.exists(output_excel):
                        logging.error(f"Output file {output_excel} was not created")
                        return jsonify({'error': 'Output file not created'}), 500
                    return jsonify({
                        'status': 'success',
                        'download_path': f'/download/output/handwritten_invoices.xlsx'
                    })
                else:
                    logging.error("Failed to process handwritten invoices")
                    return jsonify({'error': 'Failed to process handwritten invoices'}), 500
            else:
                logging.error("No handwritten invoice texts extracted")
                return jsonify({'error': 'No handwritten invoice texts extracted'}), 500
        
        elif doc_type == 'contracts':
            contract_texts = []
            for file_path in saved_files:
                logging.info(f"Extracting text from {file_path}...")
                extracted_text = extract_contract_text(file_path)
                if extracted_text:
                    contract_texts.append((Path(file_path).name, extracted_text))
                else:
                    logging.error(f"Failed to extract text from {file_path}")
            
            if contract_texts:
                logging.info("Processing all contracts...")
                results = process_all_contracts(contract_texts)
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
                    output_excel = os.path.join(output_dir, "contracts.xlsx")
                    if os.path.exists(output_excel):
                        existing_df = pd.read_excel(output_excel, engine='openpyxl')
                        df = pd.concat([existing_df, df], ignore_index=True)
                    df.to_excel(output_excel, index=False, engine='openpyxl')
                    logging.info(f"Results appended to {output_excel}")
                    if not os.path.exists(output_excel):
                        logging.error(f"Output file {output_excel} was not created")
                        return jsonify({'error': 'Output file not created'}), 500
                    return jsonify({
                        'status': 'success',
                        'download_path': f'/download/output/contracts.xlsx'
                    })
                else:
                    logging.error("Failed to process contracts")
                    return jsonify({'error': 'Failed to process contracts'}), 500
            else:
                logging.error("No contract texts extracted")
                return jsonify({'error': 'No contract texts extracted'}), 500
    
    except Exception as e:
        logging.error(f"Error processing files: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500
    finally:
        if 'session_upload_dir' in locals():
            for file_path in saved_files:
                try:
                    os.remove(file_path)
                    logging.info(f"Deleted temporary file {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to delete {file_path}: {e}")
            try:
                shutil.rmtree(session_upload_dir)
                logging.info(f"Deleted session directory {session_upload_dir}")
            except Exception as e:
                logging.warning(f"Failed to delete session directory {session_upload_dir}: {e}")

@bp.route('/download/output/<filename>')
def download_file(filename):
    """Serve the generated Excel file from the output folder for download without caching."""
    try:
        if not filename.endswith('.xlsx'):
            logging.error(f"Invalid file extension: {filename}")
            return jsonify({'error': 'Invalid file extension'}), 400
        
        core_files = ['invoices.xlsx', 'handwritten_invoices.xlsx', 'contracts.xlsx']
        
        if filename not in core_files and not filename.startswith('payslip_'):
            logging.error(f"Invalid file requested: {filename}")
            return jsonify({'error': 'Invalid file requested'}), 400
        
        file_path = os.path.join('output', filename)
        logging.info(f"Attempting to serve file: {file_path}")
        
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        if filename.startswith('payslip_'):
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
            if not validate_excel_columns(file_path, fixed_columns):
                logging.warning(f"Invalid column structure in {file_path}: skipping file")
                return jsonify({'error': f'File {filename} has invalid structure'}), 400
        
        mtime = os.path.getmtime(file_path)
        mtime_str = time.ctime(mtime)
        with open(file_path, 'rb') as f:
            etag = hashlib.md5(f.read()).hexdigest()
        logging.info(f"Serving file {file_path} for download, last modified: {mtime_str}, ETag: {etag}")
        
        response = send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            etag=etag
        )
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        logging.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': f'Download error: {str(e)}'}), 500

@bp.route('/list_files', methods=['GET'])
def list_files():
    """Return a list of all available Excel files in the output directory with valid column structure."""
    try:
        output_dir = os.path.join('output')
        files = {
            'pay_slips': [],
            'invoices': None,
            'handwritten_invoices': None,
            'contracts': None
        }

        if not os.path.exists(output_dir):
            logging.info("Output directory does not exist")
            return jsonify({'status': 'success', 'files': files})

        for file_path in Path(output_dir).glob('*.xlsx'):
            filename = file_path.name
            if filename == 'invoices.xlsx' and os.path.isfile(file_path):
                files['invoices'] = filename
            elif filename == 'handwritten_invoices.xlsx' and os.path.isfile(file_path):
                files['handwritten_invoices'] = filename
            elif filename == 'contracts.xlsx' and os.path.isfile(file_path):
                files['contracts'] = filename
            elif filename.startswith('payslip_') and os.path.isfile(file_path):
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
                if validate_excel_columns(file_path, fixed_columns):
                    client_name = normalize_name(filename.replace('payslip_', '').replace('.xlsx', ''))
                    files['pay_slips'].append({
                        'filename': filename,
                        'client_name': client_name
                    })
                else:
                    logging.warning(f"Invalid column structure in {file_path}: skipping file")
                    # Avoid deleting the file to preserve valid data

        logging.info(f"Listed files: {files}")
        return jsonify({'status': 'success', 'files': files})

    except Exception as e:
        logging.error(f"Error listing files: {str(e)}")
        return jsonify({'error': f'Error listing files: {str(e)}'}), 500

@bp.route('/reset_file', methods=['POST'])
def reset_file():
    try:
        data = request.get_json()
        file_type = data.get('file_type')
        
        if file_type not in ['invoices', 'handwritten_invoices', 'contracts']:
            return jsonify({'status': 'error', 'error': 'Invalid file type'}), 400
        
        file_name = f"{file_type}.xlsx"
        file_path = os.path.join('output', file_name)
        
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'error': f'{file_name} does not exist'}), 404
        
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.value = None
        workbook.save(file_path)
        
        return jsonify({'status': 'success', 'message': f'{file_name} reset successfully'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@bp.route('/delete_payslip', methods=['POST'])
def delete_payslip():
    """Delete a specific payslip file from the output directory."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename or not filename.startswith('payslip_') or not filename.endswith('.xlsx'):
            logging.error(f"Invalid payslip filename: {filename}")
            return jsonify({'status': 'error', 'error': 'Invalid payslip filename'}), 400
        
        file_path = os.path.join('output', filename)
        
        if not os.path.exists(file_path):
            logging.error(f"Payslip file not found: {file_path}")
            return jsonify({'status': 'error', 'error': f'Payslip file {filename} does not exist'}), 404
        
        os.remove(file_path)
        logging.info(f"Deleted payslip file: {file_path}")
        
        return jsonify({'status': 'success', 'message': f'{filename} deleted successfully'})
    
    except Exception as e:
        logging.error(f"Error deleting payslip file {filename}: {str(e)}")
        return jsonify({'status': 'error', 'error': f'Error deleting {filename}: {str(e)}'}), 500