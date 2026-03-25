"""
Streamlit app: Extract text from image/CSV/PDF/paste → LLM → Output → Download
"""

import io
import json
import os
import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
from google import genai
import pypdf


# ── 1. LLM CALLS ─────────────────────────────────────────────────────────────

def call_llm_csv(text: str) -> str:
    """Ask Gemini to extract structured data as a JSON array for export."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = f"""
    Extract all structured data from the text below and return it as a JSON array
    of flat objects (same keys in every object). Return ONLY valid JSON, no explanation.

    Text:
    {text}
    """
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text


def call_llm_slide(text: str) -> str:
    """Ask Gemini to reformat content as clean slide-ready bullet points."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = f"""
    Reformat the content below into clean, concise bullet points suitable for a
    presentation slide. Group related points under short bold headings where helpful.
    Return plain text only, no JSON, no markdown code blocks.

    Text:
    {text}
    """
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text


def call_llm_summary(text: str) -> str:
    """Ask Gemini to produce a summary and action items as JSON."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = f"""
    Read the text below and return a JSON object with exactly two keys:
    - "summary": a list of 3-5 strings, each a key takeaway from the text
    - "action_items": a list of objects with keys "task", "owner" (if mentioned, else "Unassigned"), and "due_date" (if mentioned, else "TBD")

    Return ONLY valid JSON, no explanation.

    Text:
    {text}
    """
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text


# ── 2. TEXT EXTRACTION HELPERS ───────────────────────────────────────────────

def extract_text_from_image(uploaded_file) -> str:
    """Use pytesseract (OCR) to pull text out of an uploaded image."""
    image = Image.open(uploaded_file)
    return pytesseract.image_to_string(image)


def extract_text_from_csv(uploaded_file) -> str:
    """Read a CSV and convert it to plain text for the LLM."""
    df = pd.read_csv(uploaded_file)
    return df.to_string(index=False)


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from each page of an uploaded PDF."""
    reader = pypdf.PdfReader(uploaded_file)
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n\n".join(pages)


# ── 3. RESPONSE PARSERS ──────────────────────────────────────────────────────

def parse_csv_response(response: str) -> pd.DataFrame:
    """Parse a JSON array from the LLM into a DataFrame."""
    cleaned = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of objects.")
    return pd.DataFrame(data)


def parse_summary_response(response: str) -> dict:
    """Parse the summary + action items JSON from the LLM."""
    cleaned = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(cleaned)
    if "summary" not in data or "action_items" not in data:
        raise ValueError("Response missing 'summary' or 'action_items' keys.")
    return data


# ── 4. DOWNLOAD HELPERS ──────────────────────────────────────────────────────

def download_buttons(df: pd.DataFrame, csv_filename: str, excel_filename: str):
    """Show both a CSV and an Excel download button for a given DataFrame."""
    col1, col2 = st.columns(2)

    with col1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv_bytes,
            file_name=csv_filename,
            mime="text/csv"
        )

    with col2:
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine="openpyxl")
        st.download_button(
            label="⬇️ Download as Excel",
            data=excel_buffer.getvalue(),
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ── 5. STREAMLIT UI ──────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Navistone Content Extractor", layout="centered")
    st.title("📋 Navistone Content Extractor")
    st.caption("Upload an image, PDF, CSV, or paste text — then choose how to extract it.")

    # ── Input section ────────────────────────────────────────────────────────
    st.header("1 · Provide input")
    input_mode = st.radio("Input type", ["Image (OCR)", "PDF", "CSV", "Paste text"], horizontal=True)

    raw_text = ""

    if input_mode == "Image (OCR)":
        img_file = st.file_uploader("Upload PNG or JPG", type=["png", "jpg", "jpeg"])
        if img_file:
            st.image(img_file, caption="Uploaded image", use_container_width=True)
            with st.spinner("Reading text from image..."):
                raw_text = extract_text_from_image(img_file)
            st.text_area("Extracted text (OCR)", raw_text, height=160, disabled=True)

    elif input_mode == "PDF":
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf_file:
            with st.spinner("Reading text from PDF..."):
                raw_text = extract_text_from_pdf(pdf_file)
            st.text_area("Extracted text (PDF)", raw_text, height=160, disabled=True)

    elif input_mode == "CSV":
        csv_file = st.file_uploader("Upload CSV", type=["csv"])
        if csv_file:
            raw_text = extract_text_from_csv(csv_file)
            st.text_area("CSV as text", raw_text, height=160, disabled=True)

    else:  # Paste text
        raw_text = st.text_area("Paste your text here", height=200,
                                placeholder="Paste an email, meeting notes, campaign details…")

    # ── Extraction mode ──────────────────────────────────────────────────────
    st.header("2 · Choose extraction type")
    extraction_mode = st.radio(
        "What do you want?",
        ["Extract to CSV", "Slide Ready", "Summary & Action Items"],
        horizontal=True,
        help=(
            "Extract to CSV: pulls structured data into a downloadable spreadsheet. "
            "Slide Ready: formats content as bullet points for a presentation. "
            "Summary & Action Items: key takeaways plus a list of tasks and owners."
        )
    )

    # ── Run ──────────────────────────────────────────────────────────────────
    st.header("3 · Extract")
    run = st.button("🚀 Run", disabled=not raw_text.strip())

    if run:
        with st.spinner("Sending to Gemini..."):
            try:
                if extraction_mode == "Extract to CSV":
                    llm_output = call_llm_csv(raw_text)
                elif extraction_mode == "Slide Ready":
                    llm_output = call_llm_slide(raw_text)
                else:
                    llm_output = call_llm_summary(raw_text)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        # ── Display results ──────────────────────────────────────────────────
        st.header("4 · Results")

        if extraction_mode == "Extract to CSV":
            try:
                df = parse_csv_response(llm_output)
                st.dataframe(df, use_container_width=True)
                download_buttons(df, "extracted_data.csv", "extracted_data.xlsx")
            except (json.JSONDecodeError, ValueError) as e:
                st.error(f"Could not parse response: {e}")
                st.code(llm_output)

        elif extraction_mode == "Slide Ready":
            st.markdown(llm_output)
            st.download_button("⬇️ Download as .txt", data=llm_output.encode("utf-8"),
                               file_name="slide_content.txt", mime="text/plain")

        else:  # Summary & Action Items
            try:
                data = parse_summary_response(llm_output)

                st.subheader("📌 Key Takeaways")
                for point in data["summary"]:
                    st.markdown(f"- {point}")

                st.subheader("✅ Action Items")
                if data["action_items"]:
                    action_df = pd.DataFrame(data["action_items"])
                    st.dataframe(action_df, use_container_width=True)
                    download_buttons(action_df, "action_items.csv", "action_items.xlsx")
                else:
                    st.info("No action items found.")

            except (json.JSONDecodeError, ValueError) as e:
                st.error(f"Could not parse response: {e}")
                st.code(llm_output)


if __name__ == "__main__":
    main()
