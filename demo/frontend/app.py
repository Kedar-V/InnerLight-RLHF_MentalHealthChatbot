import os
import streamlit as st
import requests

st.set_page_config(page_title="RLHF Demo", layout="wide")
st.title("RLHF Demo — Streamlit UI")

# Backend URL is configurable via env `BACKEND_URL` so the app works both inside
# Docker Compose (use `http://backend:8000`) and when running locally (use `http://localhost:8000`).
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

prompt = st.text_area("Enter a prompt", height=150)

if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        try:
            resp = requests.post(f"{BACKEND_URL}/generate", json={"prompt": prompt}, timeout=10)
            if resp.ok:
                data = resp.json()
                st.subheader("Response")
                st.write(data.get("response", "(no response)"))
            else:
                st.error(f"Backend error: {resp.status_code} - {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
