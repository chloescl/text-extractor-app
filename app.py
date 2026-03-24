"""
Streamlit app: Extract text from image/CSV/paste → LLM → DataFrame → CSV download
"""

import json
import os
import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
from google import genai


# ── 1. LLM CALL (Google Gemini - free tier) ──────────────────────────────────
def call_llm(text: str) -> str:
    """Send text to Google Gemini and return a JSON string."""

    # Reads the API key from Streamlit secrets
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = f"""
    Extract all structured data from the text below and return it as a JSON array
    of flat objects (same keys in every object). Return ONLY valid JSON, no explanation.

    Text:
    {text}
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text


# ── 2. TEXT EXTRACTION HELPERS ───────────────────────────────────────────────
def extract_text_from_image(uploaded_file) -> str:
    """Use pytesseract (OCR) to pull text out of an uploaded image."""
    image = Image.open(uploaded_file)
    return pytesseract.image_to_string(image)


def extract_text_from_csv(uploaded_file) -> str:
    """Read a CSV and convert it to a plain-text representation for the LLM."""
    df = pd.read_csv(uploaded_file)
    return df.to_string(index=False)


# ── 3. RESPONSE PARSER ───────────────────────────────────────────────────────
def parse_llm_response(response: str) -> pd.DataFrame:
    """
    Parse the LLM's JSON string into a pandas DataFrame.
    Expects a JSON array of flat objects: [{...}, {...}, ...]
    """
    # Strip markdown fences if the model wraps the JSON in ```json ... ```
    cleaned = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(cleaned)

    if not isinstance(data, list):
        raise ValueError("LLM response must be a JSON array of objects.")

    return pd.DataFrame(data)


# ── 4. STREAMLIT UI ──────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Text → LLM → Table", layout="centered")
    st.title("📋 Text → LLM → Structured Table")
    st.caption("Upload an image, a CSV, or paste text — the LLM extracts structure for you.")

    # ── Input section ────────────────────────────────────────────────────────
    st.header("1 · Provide input")
    input_mode = st.radio("Input type", ["Image (OCR)", "CSV", "Paste text"], horizontal=True)

    raw_text = ""

    if input_mode == "Image (OCR)":
        img_file = st.file_uploader("Upload PNG or JPG", type=["png", "jpg", "jpeg"])
        if img_file:
            st.image(img_file, caption="Uploaded image", use_container_width=True)
            raw_text = extract_text_from_image(img_file)
            st.text_area("Extracted text (OCR)", raw_text, height=160, disabled=True)

    elif input_mode == "CSV":
        csv_file = st.file_uploader("Upload CSV", type=["csv"])
        if csv_file:
            raw_text = extract_text_from_csv(csv_file)
            st.text_area("CSV as text", raw_text, height=160, disabled=True)

    else:  # Paste text
        raw_text = st.text_area("Paste your text here", height=200,
                                placeholder="Names, dates, addresses, table data…")

    # ── LLM call ─────────────────────────────────────────────────────────────
    st.header("2 · Extract structured data")
    run = st.button("🚀 Send to LLM", disabled=not raw_text.strip())

    if run:
        with st.spinner("Calling LLM…"):
            try:
                llm_output = call_llm(raw_text)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        with st.expander("Raw LLM response", expanded=False):
            st.code(llm_output, language="json")

        # ── Parse & display ──────────────────────────────────────────────────
        st.header("3 · Result table")
        try:
            df = parse_llm_response(llm_output)
            st.dataframe(df, use_container_width=True)

            # ── CSV download ─────────────────────────────────────────────────
            st.header("4 · Download")
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv_bytes,
                file_name="extracted_data.csv",
                mime="text/csv",
            )

        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Could not parse LLM response: {e}")
            st.info("Make sure your LLM call returns a JSON array of flat objects.")


if __name__ == "__main__":
    main()
