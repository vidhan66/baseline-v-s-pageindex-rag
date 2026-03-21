# baseline-v-s-pageindex-rag

## Streamlit comparison app

This repo now includes an `app` directory with a simple Streamlit UI that:
- uploads one PDF,
- indexes it for both Baseline RAG and PageIndex RAG,
- asks one question and gets both answers side-by-side,
- shows token usage for both strategies,
- keeps simple conversation buffer memory per strategy.

### Run

1. Install dependencies:
   - `pip install -r app/requirements.txt`
2. Set env vars in `.env`:
   - `OPENAI_API_KEY=...`
   - `PAGEINDEX_API_KEY=...`
3. Start app:
   - `streamlit run app/streamlit_app.py`