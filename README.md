# PolymerGPT
PolymerGPT: Predictive Modeling of Polymer Properties Using LLMs


🧪 PolymerGPT: Polymer Similarity & Property Explorer
A lightweight, interactive Streamlit application for exploring and comparing polymers based on their SMILES strings using pretrained transformer embeddings (e.g., polyBERT). Given a polymer's pseudo-SMILES (pSMILES) representation, the tool searches for similar polymers in a dataset and optionally filters results by glass transition temperature (Tg).

🚀 Features
🔍 pSMILES-Based Similarity Search: Uses cosine similarity between sentence embeddings to find chemically similar polymers.

🤖 Pretrained Transformer Support: Compatible with any SentenceTransformer model (e.g., kuelumbus/polyBERT).

📊 Interactive Results Filtering: Set similarity thresholds and Tg windows from the sidebar.

📁 Fast Parquet File Handling: Reads up to 10,000 polymer entries from a parquet file for quick searches.

⚡ Streamlit-Powered Interface: Runs in-browser with responsive, intuitive UI.

🛠️ Requirements
pip install streamlit pandas sentence-transformers scikit-learn
You also need a local or remote .parquet file of polymers with at least a smiles column (optionally Tg for filtering).

📂 File Structure
PolymerGPT.py: Main Streamlit application file.

🧠 Usage
1. Run the App
streamlit run PolymerGPT.py

3. Inputs (via UI)
📍 Parquet File Path: Local path to a .parquet file containing polymer data.

🤖 Model Name: Any sentence-transformers compatible model. Default: kuelumbus/polyBERT.

📈 Cosine Similarity Threshold: Filter results based on closeness to query polymer.

🌡️ Tg Window (°C): Optional Tg-based filtering.

3. Query
Enter a pSMILES string in the main input box (e.g., [✱]CC[✱]) and view similar polymers with metadata.

🧪 Example Dataset Format
Your Parquet file should contain at least:

smiles	Tg	...
[✱]CC[✱]	100.0	...
[✱]C(C)C[✱]	80.0	...

📊 Output
Displays a table of polymers most similar to your input pSMILES.

Each result includes:

SMILES

Similarity Score

Optional Tg

Other columns from the input data.

📌 Notes
By default, only the first 10,000 entries are processed to optimize performance.

The app caches embeddings and model loading to speed up repeated use.

