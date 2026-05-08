"""
AI-Assisted Bid Document Extraction Tool

Run:
python script.py
"""

import os
import sys
import json
import time
import html
from typing import List, Dict, Any
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from concurrent.futures import ThreadPoolExecutor, as_completed

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.units import inch


# -----------------------------
# Settings to reduce runtime
# -----------------------------

MODEL_NAME = "gpt-5.1"
MAX_CHARS_PER_CHUNK = 5000
MAX_WORKERS = 3
REQUEST_TIMEOUT = 90

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_text_with_pymupdf(pdf_path: str) -> List[Dict[str, Any]]:
    doc = pymupdf.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append({
            "page": page_num,
            "text": text.strip()
        })

    return pages


def extract_text_with_ocr(pdf_path: str) -> List[Dict[str, Any]]:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        raise ImportError(
            "OCR requires pdf2image and pytesseract. Install with: "
            "pip install pdf2image pytesseract pillow"
        )

    images = convert_from_path(pdf_path)
    pages = []

    for i, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image)
        pages.append({
            "page": i,
            "text": text.strip()
        })

    return pages


def extract_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    pages = extract_text_with_pymupdf(pdf_path)
    total_text = " ".join(page["text"] for page in pages)

    if len(total_text.strip()) < 500:
        print("PDF text extraction was weak. Trying OCR fallback...")
        pages = extract_text_with_ocr(pdf_path)

    return pages


def chunk_pages(pages: List[Dict[str, Any]], max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    chunks = []
    current_chunk = ""

    for page in pages:
        if not page["text"].strip():
            continue

        page_text = f"\n\n[Page {page['page']}]\n{page['text']}"

        if len(current_chunk) + len(page_text) > max_chars:
            if current_chunk.strip():
                chunks.append(current_chunk)
            current_chunk = page_text
        else:
            current_chunk += page_text

    if current_chunk.strip():
        chunks.append(current_chunk)

    return chunks


def get_json_schema(schema_name: str) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "name": schema_name,
        "schema": {
            "type": "object",
            "properties": {
                "bid_summary": {"type": "string"},
                "project_scope": {"type": "string"},
                "key_dates": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "submission_requirements": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "legal_contractual_insurance_compliance": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "ambiguities_risks_missing_info": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": [
                "bid_summary",
                "project_scope",
                "key_dates",
                "submission_requirements",
                "legal_contractual_insurance_compliance",
                "ambiguities_risks_missing_info"
            ],
            "additionalProperties": False
        },
        "strict": True
    }


def call_llm_for_chunk(chunk_text: str, chunk_number: int) -> Dict[str, Any]:
    prompt = f"""
You are analyzing a bid/RFP/procurement document.

Use only the provided document text. Do not invent facts.

Extract:
1. Plain-English bid summary
2. Project scope
3. Key dates and submission requirements
4. Legal, contractual, insurance, and compliance callouts
5. Ambiguities, risks, or missing information the bidder should review

Include page references whenever possible.

Document chunk #{chunk_number}:
{chunk_text}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        timeout=REQUEST_TIMEOUT,
        text={
            "format": get_json_schema("bid_chunk_extraction")
        }
    )

    return json.loads(response.output_text)


def merge_results(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(chunk_results) == 1:
        return chunk_results[0]

    prompt = f"""
You are combining extracted information from multiple chunks of one bid document.

Create one final clean extraction.

Rules:
- Remove duplicates.
- Keep page references where available.
- Use plain English.
- Do not invent missing information.
- If something important is missing or unclear, include it under risks or missing information.

Chunk extractions:
{json.dumps(chunk_results, indent=2)}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        timeout=REQUEST_TIMEOUT,
        text={
            "format": get_json_schema("final_bid_extraction")
        }
    )

    return json.loads(response.output_text)


def write_json_report(result: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def write_markdown_report(result: Dict[str, Any], output_path: str) -> None:
    def bullet_list(items):
        if not items:
            return "- Not found\n"
        return "\n".join(f"- {item}" for item in items) + "\n"

    markdown = f"""# AI-Assisted Bid Document Extraction Report

## 1. Plain-English Bid Summary

{result.get("bid_summary", "Not found")}

## 2. Project Scope

{result.get("project_scope", "Not found")}

## 3. Key Dates

{bullet_list(result.get("key_dates", []))}

## 4. Submission Requirements

{bullet_list(result.get("submission_requirements", []))}

## 5. Legal, Contractual, Insurance, and Compliance Callouts

{bullet_list(result.get("legal_contractual_insurance_compliance", []))}

## 6. Ambiguities, Risks, or Missing Information

{bullet_list(result.get("ambiguities_risks_missing_info", []))}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)


def write_pdf_report(result: Dict[str, Any], output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        spaceAfter=16
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14,
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        "CustomBullet",
        parent=styles["BodyText"],
        fontSize=10,
        leading=13,
        leftIndent=12
    )

    story = []

    def safe_text(text):
        return html.escape(str(text)) if text else "Not found"

    def add_heading(text):
        story.append(Paragraph(safe_text(text), heading_style))

    def add_paragraph(text):
        story.append(Paragraph(safe_text(text), body_style))
        story.append(Spacer(1, 6))

    def add_bullets(items):
        if not items:
            add_paragraph("Not found")
            return

        bullet_items = [
            ListItem(Paragraph(safe_text(item), bullet_style))
            for item in items
        ]

        story.append(ListFlowable(
            bullet_items,
            bulletType="bullet",
            leftIndent=18
        ))
        story.append(Spacer(1, 8))

    story.append(Paragraph("AI-Assisted Bid Document Extraction Report", title_style))

    add_heading("1. Plain-English Bid Summary")
    add_paragraph(result.get("bid_summary", "Not found"))

    add_heading("2. Project Scope")
    add_paragraph(result.get("project_scope", "Not found"))

    add_heading("3. Key Dates")
    add_bullets(result.get("key_dates", []))

    add_heading("4. Submission Requirements")
    add_bullets(result.get("submission_requirements", []))

    add_heading("5. Legal, Contractual, Insurance, and Compliance Callouts")
    add_bullets(result.get("legal_contractual_insurance_compliance", []))

    add_heading("6. Ambiguities, Risks, or Missing Information")
    add_bullets(result.get("ambiguities_risks_missing_info", []))

    doc.build(story)


def analyze_chunks_in_parallel(chunks: List[str]) -> List[Dict[str, Any]]:
    results = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(call_llm_for_chunk, chunk, i + 1): i
            for i, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            chunk_number = index + 1

            try:
                results[index] = future.result()
                print(f"Chunk {chunk_number} completed.")
            except RateLimitError:
                print("\nOpenAI API quota or rate limit issue.")
                print("Check billing and usage limits in your OpenAI account.")
                sys.exit(1)
            except Exception as e:
                print(f"\nError analyzing chunk {chunk_number}: {e}")
                sys.exit(1)

    return results


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY. Add it to your .env file.")
        sys.exit(1)

    root = Tk()
    root.withdraw()

    pdf_path = askopenfilename(
        title="Select a Bid PDF Document",
        filetypes=[("PDF Files", "*.pdf")]
    )

    if not pdf_path:
        print("No PDF selected.")
        sys.exit(1)

    overall_start = time.time()

    print(f"Selected file: {pdf_path}")

    print("Extracting text from PDF...")
    pages = extract_pdf_text(pdf_path)

    print("Chunking document...")
    chunks = chunk_pages(pages)

    if not chunks:
        print("No readable text found in the PDF.")
        sys.exit(1)

    print(f"Analyzing {len(chunks)} chunk(s) with LLM...")
    print(f"Runtime settings: chunk size={MAX_CHARS_PER_CHUNK}, parallel workers={MAX_WORKERS}, timeout={REQUEST_TIMEOUT}s")

    chunk_results = analyze_chunks_in_parallel(chunks)

    print("Merging results...")
    final_result = merge_results(chunk_results)

    json_output = "bid_extraction_data.json"
    markdown_output = "bid_extraction_report.md"
    pdf_output = "bid_extraction_report.pdf"

    write_json_report(final_result, json_output)
    write_markdown_report(final_result, markdown_output)
    write_pdf_report(final_result, pdf_output)

    total_time = round(time.time() - overall_start, 2)

    print("\nDone.")
    print(f"Completed in {total_time} seconds.")
    print("Generated files:")
    print(f"- {json_output}")
    print(f"- {markdown_output}")
    print(f"- {pdf_output}")


if __name__ == "__main__":
    main()
