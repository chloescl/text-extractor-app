"""
Streamlit app: Extract text from image/CSV/XLSX/PDF/paste → LLM → Output → Download
Supports uploading up to 10 files at once.
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


def call_llm_slide(text: str, num_slides: int, slide_titles: str) -> str:
    """Ask Gemini to reformat content as structured slide-ready bullet points."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    if slide_titles.strip():
        titles = [t.strip() for t in slide_titles.split(",") if t.strip()]
        title_instruction = f"Use exactly these slide titles in this order: {', '.join(titles)}."
    else:
        title_instruction = f"Create {num_slides} slides with appropriate titles based on the content."
    prompt = f"""
    Reformat the content below into structured presentation slides.
    {title_instruction}
    Each slide should have a clear bold title followed by 3-5 concise bullet points.
    Distribute the content evenly and logically across the slides.
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


def extract_text_from_xlsx(uploaded_file) -> str:
    """Read an Excel file and convert it to plain text for the LLM."""
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    return df.to_string(index=False)


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from each page of an uploaded PDF."""
    reader = pypdf.PdfReader(uploaded_file)
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n\n".join(pages)


def extract_text_from_file(uploaded_file, input_mode: str) -> str:
    """Route a single file to the correct extractor based on input mode."""
    if input_mode == "Image (OCR)":
        return extract_text_from_image(uploaded_file)
    elif input_mode == "PDF":
        return extract_text_from_pdf(uploaded_file)
    elif input_mode == "CSV":
        return extract_text_from_csv(uploaded_file)
    elif input_mode == "Excel (.xlsx)":
        return extract_text_from_xlsx(uploaded_file)
    return ""


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


# ── 4. STREAMLIT UI ──────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Navistone Content Extractor", layout="centered")
    st.title("📋 Navistone Content Extractor")
    st.caption("Upload up to 10 files or paste text — then choose how to extract it.")

    # ── Input section ────────────────────────────────────────────────────────
    st.header("1 · Provide input")
    input_mode = st.radio("Input type", ["Image (OCR)", "PDF", "CSV", "Excel (.xlsx)", "Paste text"], horizontal=True)

    raw_text = ""

    if input_mode == "Paste text":
        raw_text = st.text_area("Paste your text here", height=200,
                                placeholder="Paste an email, meeting notes, campaign details…")
    else:
        # Map input mode to accepted file types
        file_type_map = {
            "Image (OCR)": ["png", "jpg", "jpeg"],
            "PDF":         ["pdf"],
            "CSV":         ["csv"],
            "Excel (.xlsx)": ["xlsx"],
        }
        accepted_types = file_type_map[input_mode]

        uploaded_files = st.file_uploader(
            f"Upload up to 10 {input_mode} files",
            type=accepted_types,
            accept_multiple_files=True,
        )

        # Enforce 10 file limit
        if uploaded_files and len(uploaded_files) > 10:
            st.warning("Please upload a maximum of 10 files. Only the first 10 will be used.")
            uploaded_files = uploaded_files[:10]

        if uploaded_files:
            all_texts = []
            for f in uploaded_files:
                with st.spinner(f"Reading {f.name}..."):
                    try:
                        text = extract_text_from_file(f, input_mode)
                        all_texts.append(f"--- {f.name} ---\n{text}")
                    except Exception as e:
                        st.error(f"Could not read {f.name}: {e}")

            if all_texts:
                # Combine all file contents into one block of text for the LLM
                raw_text = "\n\n".join(all_texts)
                st.success(f"✅ {len(all_texts)} file(s) loaded successfully.")
                with st.expander("Preview extracted text", expanded=False):
                    st.text_area("Combined text from all files", raw_text, height=200, disabled=True)

    # ── Extraction mode ──────────────────────────────────────────────────────
    st.header("2 · Choose extraction type")
    extraction_mode = st.radio(
        "What do you want?",
        ["Extract to CSV", "Extract to Excel", "Slide Ready", "Summary & Action Items"],
        horizontal=True,
    )

    # ── Slide sub-options ────────────────────────────────────────────────────
    num_slides = 3
    slide_titles = ""

    if extraction_mode == "Slide Ready":
        st.markdown("**Slide options**")
        col1, col2 = st.columns([1, 2])
        with col1:
            num_slides = st.number_input(
                "Number of slides",
                min_value=1, max_value=20, value=3, step=1,
                help="How many slides to split the content across."
            )
        with col2:
            slide_titles = st.text_input(
                "Slide titles (optional)",
                placeholder="e.g. Overview, Campaign Details, Next Steps",
                help="Enter comma-separated titles. If left blank, titles will be auto-generated."
            )
        st.caption("💡 Tip: if you enter titles, the number of slides will match the number of titles you provide.")

    # ── Run ──────────────────────────────────────────────────────────────────
    st.header("3 · Extract")
    run = st.button("🚀 Run", disabled=not raw_text.strip())

    if run:
        with st.spinner("Sending to Gemini..."):
            try:
                if extraction_mode in ("Extract to CSV", "Extract to Excel"):
                    llm_output = call_llm_csv(raw_text)
                elif extraction_mode == "Slide Ready":
                    llm_output = call_llm_slide(raw_text, num_slides, slide_titles)
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
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download as CSV", data=csv_bytes,
                                   file_name="extracted_data.csv", mime="text/csv")
            except (json.JSONDecodeError, ValueError) as e:
                st.error(f"Could not parse response: {e}")
                st.code(llm_output)

        elif extraction_mode == "Extract to Excel":
            try:
                df = parse_csv_response(llm_output)
                st.dataframe(df, use_container_width=True)
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False, engine="openpyxl")
                st.download_button("⬇️ Download as Excel", data=excel_buffer.getvalue(),
                                   file_name="extracted_data.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
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
                    excel_buffer = io.BytesIO()
                    action_df.to_excel(excel_buffer, index=False, engine="openpyxl")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("⬇️ Download as CSV",
                                           data=action_df.to_csv(index=False).encode("utf-8"),
                                           file_name="action_items.csv", mime="text/csv")
                    with col2:
                        st.download_button("⬇️ Download as Excel",
                                           data=excel_buffer.getvalue(),
                                           file_name="action_items.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.info("No action items found.")

            except (json.JSONDecodeError, ValueError) as e:
                st.error(f"Could not parse response: {e}")
                st.code(llm_output)


if __name__ == "__main__":
    main()
