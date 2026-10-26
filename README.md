# 🧬 Glimpse
### Explainable early detection of rare metabolic diseases from LC-HR-MS spectra.

> *"Rare diseases are not rare. They are just rarely diagnosed."*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://glimpse-utewutpspwnhkaigecpk8f.streamlit.app/)
[![CI](https://github.com/MOKOOLI/glimpse/actions/workflows/ci.yml/badge.svg)](https://github.com/MOKOOLI/glimpse/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 The Problem

Rare metabolic disorders often take months to years to diagnose. Mass spectrometry (LC-HR-MS) carries the signal — but the data is noisy, high-dimensional, and hard to interpret. Every day of delay can mean irreversible neurological damage.

## 💡 The Solution

**Glimpse** is an end-to-end, explainable machine learning pipeline that turns raw spectra into a risk score **plus** a human-readable explanation of *which* features drove that decision.

### ✨ Key Features
- **Explainable AI (XAI):** Every prediction is accompanied by SHAP-based feature attributions. No black boxes.
- **Honest Evaluation:** Patient-level splitting (GroupKFold) to prevent data leakage.
- **Production-Ready:** REST API (FastAPI), Interactive Dashboard (Streamlit), Docker support, and CI/CD.
- **Workflow Automation:** Integrated n8n workflow to send real-time alerts for high-risk cases.

---

## 🚀 Live Demo

👉 **[Try the live app here](https://glimpse-utewutpspwnhkaigecpk8f.streamlit.app/)** 👈

*(Note: The live demo uses **synthetic data** to demonstrate the end-to-end pipeline. The architecture is ready to be trained on real clinical data. See the [Disclaimer](#️-disclaimer--dataset) below.)*

---

## 🏗️ Architecture

```text
Raw mzML/mzXML (or Tabular Peaks)
   │
   ▼
[Preprocessing]  peak picking → baseline correction → normalization
   │
   ▼
[Features]       peak intensity matrix + derived ratios + QC flags
   │
   ▼
[Model]          XGBoost + SHAP (patient-level cross-validation)
   │
   ▼
[Outputs]        risk score + top contributing features
   │
   ├── REST API (FastAPI)
   ├── Dashboard (Streamlit)
   ├── Batch CLI
   └── n8n alert workflow