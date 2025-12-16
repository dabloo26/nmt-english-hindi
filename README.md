# 🇮🇳 English → Hindi Neural Machine Translation

A Streamlit web app for translating English text to Hindi using a Transformer model.

## 📁 Project Structure

```
nmt-english-hindi/
├── app.py                          ← Main Streamlit app
├── requirements.txt                ← Python dependencies
├── packages.txt                    ← System dependencies
├── README.md                       ← This file
└── models/                         ← ⚠️ ADD YOUR FILES HERE
    ├── transformer_model.keras     ← Download from Google Drive
    ├── en_tokenizer.model          ← Download from Google Drive
    └── hi_tokenizer.model          ← Download from Google Drive
```

## 🔧 Setup Instructions

### Prerequisites
- Python 3.8-3.12 (Python 3.13 has compatibility issues with some dependencies)
- pip package manager

### Step 1: Download Model Files from Google Drive

Download these 3 files from your Google Drive:

| File | Location in Google Drive |
|------|--------------------------|
| `transformer_model.keras` | `/NMT_English_Hindi/models/transformer/transformer_model.keras` |
| `en_tokenizer.model` | `/NMT_English_Hindi/tokenizers/en_tokenizer.model` |
| `hi_tokenizer.model` | `/NMT_English_Hindi/tokenizers/hi_tokenizer.model` |

### Step 2: Add Files to `models/` Folder

Put all 3 downloaded files into the `models/` folder:

```
models/
├── transformer_model.keras
├── en_tokenizer.model
└── hi_tokenizer.model
```

### Step 3: Run Locally (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🚀 Deploy to Streamlit Cloud

### Option A: GitHub + Streamlit Cloud (Recommended)

1. **Create a GitHub repository**
   - Go to github.com and create a new repo named `nmt-english-hindi`

2. **Upload all files** (including files in `models/` folder)
   
   ⚠️ **Note:** If `transformer_model.keras` is too large (>100MB), use Git LFS:
   ```bash
   git lfs install
   git lfs track "*.keras"
   git add .gitattributes
   git add .
   git commit -m "Initial commit"
   git push
   ```

3. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repo
   - Set main file path: `app.py`
   - Click "Deploy"

### Option B: Using Git LFS for Large Files

If your model file is >100MB:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.keras"
git lfs track "*.model"

# Add and commit
git add .gitattributes
git add .
git commit -m "Add model files with LFS"
git push
```

## 📊 Model Information

| Metric | Value |
|--------|-------|
| Architecture | Transformer (4 layers, 256 dim, 8 heads) |
| Training Data | IIT Bombay Corpus (310K pairs) |
| BLEU Score | 16.48 |
| chrF Score | 38.96 |
| Vocabulary | 8K tokens (BPE) |

## 🔄 Translation Pipeline

1. **Input:** English sentence
2. **Preprocessing:** Lowercase, clean text
3. **Reordering:** Convert SVO → SOV (to match Hindi word order)
4. **Tokenization:** BPE encoding
5. **Translation:** Transformer model (greedy decoding)
6. **Output:** Hindi sentence

## 📝 Example Translations

| English | Hindi |
|---------|-------|
| The boy eats an apple. | लड़का एक सेब खाता है। |
| India is a beautiful country. | भारत एक सुंदर देश है। |
| I love my family. | मैं अपने परिवार से प्यार करता हूं। |

## ⚠️ Limitations

- Works best with short/medium sentences (≤20 words)
- May struggle with complex grammar or idioms
- First load takes ~30 seconds (loading model + Stanza)

## 🛠️ Troubleshooting

**Error: Model file not found**
- Make sure all 3 files are in the `models/` folder

**Error: Memory issues on Streamlit Cloud**
- The free tier has limited memory; model loading may fail
- Consider using a smaller model or paid tier

**Slow first translation**
- First translation downloads Stanza English model (~100MB)
- Subsequent translations are faster

## 📄 License

This project was created for ENPM665 NLP Course at University of Maryland.
