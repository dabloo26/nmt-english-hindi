# ============================================================================
# NMT English-Hindi Translation App (Streamlit)
# ============================================================================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Layer, Embedding, Dense, Dropout, 
    LayerNormalization, MultiHeadAttention
)
import sentencepiece as spm
import os

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="English → Hindi Translator",
    page_icon="🇮🇳",
    layout="centered"
)

# ============================================================================
# 1. TOKENIZER CLASS
# ============================================================================

class BPETokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
    
    @property
    def bos_id(self):
        return self.sp.bos_id()
    
    @property
    def eos_id(self):
        return self.sp.eos_id()
    
    @property
    def pad_id(self):
        return self.sp.pad_id()
    
    def encode(self, text, max_len=50):
        ids = [self.bos_id] + self.sp.Encode(text) + [self.eos_id]
        if len(ids) > max_len:
            ids = ids[:max_len-1] + [self.eos_id]
        ids = ids + [self.pad_id] * (max_len - len(ids))
        return np.array([ids])
    
    def decode(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        eos = self.eos_id
        ids = [int(i) for i in ids if i > 0 and i != eos]
        return self.sp.Decode(ids)

# ============================================================================
# 2. SOURCE REORDERING (SVO → SOV)
# ============================================================================

@st.cache_resource
def load_stanza():
    """Load Stanza for English parsing."""
    import stanza
    try:
        stanza.download('en', processors='tokenize,mwt,pos,lemma,depparse', verbose=False)
    except:
        pass
    return stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', verbose=False)

def reorder_sentence(sentence, nlp):
    """Reorder English SVO to Hindi-like SOV."""
    try:
        doc = nlp(sentence.lower().strip())
        
        for sent in doc.sentences:
            words = sent.words
            word_list = [w.text for w in words]
            
            # Find subject, verb, object indices
            subject_idx = None
            verb_idx = None
            object_idx = None
            
            for i, w in enumerate(words):
                if w.deprel in ['nsubj', 'nsubj:pass'] and subject_idx is None:
                    subject_idx = i
                elif w.deprel == 'root' and verb_idx is None:
                    verb_idx = i
                elif w.deprel in ['obj', 'dobj', 'iobj'] and object_idx is None:
                    object_idx = i
            
            # Reorder if S < V < O
            if (subject_idx is not None and verb_idx is not None and 
                object_idx is not None and subject_idx < verb_idx < object_idx):
                
                # Swap verb and object positions
                reordered = word_list.copy()
                reordered[verb_idx] = word_list[object_idx]
                reordered[object_idx] = word_list[verb_idx]
                
                return ' '.join(reordered)
        
        return sentence.lower().strip()
    except Exception as e:
        return sentence.lower().strip()

# ============================================================================
# 3. TRANSFORMER MODEL DEFINITION
# ============================================================================

class PositionalEncoding(Layer):
    def __init__(self, max_len=50, d_model=256, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({'max_len': self.max_len, 'd_model': self.d_model})
        return config

class TransformerEncoderLayer(Layer):
    def __init__(self, d_model=256, num_heads=8, dff=512, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model // self.num_heads)
        self.ffn = keras.Sequential([Dense(self.dff, activation='relu'), Dense(self.d_model)])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)
        super().build(input_shape)
    
    def call(self, x, training=False):
        attn = self.dropout1(self.mha(x, x, x), training=training)
        x = self.norm1(x + attn)
        ffn_out = self.dropout2(self.ffn(x), training=training)
        return self.norm2(x + ffn_out)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model, 'num_heads': self.num_heads,
            'dff': self.dff, 'dropout_rate': self.dropout_rate
        })
        return config

class TransformerDecoderLayer(Layer):
    def __init__(self, d_model=256, num_heads=8, dff=512, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.mha1 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model // self.num_heads)
        self.mha2 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model // self.num_heads)
        self.ffn = keras.Sequential([Dense(self.dff, activation='relu'), Dense(self.d_model)])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)
        self.dropout3 = Dropout(self.dropout_rate)
        super().build(input_shape)
    
    def call(self, x, enc_output, training=False):
        attn1 = self.dropout1(self.mha1(x, x, x, use_causal_mask=True), training=training)
        x = self.norm1(x + attn1)
        attn2 = self.dropout2(self.mha2(x, enc_output, enc_output), training=training)
        x = self.norm2(x + attn2)
        ffn_out = self.dropout3(self.ffn(x), training=training)
        return self.norm3(x + ffn_out)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model, 'num_heads': self.num_heads,
            'dff': self.dff, 'dropout_rate': self.dropout_rate
        })
        return config

class Transformer(keras.Model):
    def __init__(self, num_layers=4, d_model=256, num_heads=8, dff=512,
                 src_vocab=8000, tgt_vocab=8000, max_len=50, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
        self.enc_embedding = Embedding(src_vocab, d_model)
        self.enc_pos = PositionalEncoding(max_len, d_model)
        self.enc_dropout = Dropout(dropout_rate)
        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate) 
                          for _ in range(num_layers)]
        
        self.dec_embedding = Embedding(tgt_vocab, d_model)
        self.dec_pos = PositionalEncoding(max_len, d_model)
        self.dec_dropout = Dropout(dropout_rate)
        self.dec_layers = [TransformerDecoderLayer(d_model, num_heads, dff, dropout_rate)
                          for _ in range(num_layers)]
        
        self.final_layer = Dense(tgt_vocab, activation='softmax', dtype='float32')
    
    def call(self, inputs, training=False):
        enc_input, dec_input = inputs
        
        x = self.enc_embedding(enc_input)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.enc_pos(x)
        x = self.enc_dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training)
        enc_output = x
        
        y = self.dec_embedding(dec_input)
        y = y * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        y = self.dec_pos(y)
        y = self.dec_dropout(y, training=training)
        for layer in self.dec_layers:
            y = layer(y, enc_output, training=training)
        
        return self.final_layer(y)

# ============================================================================
# 4. LOAD MODEL AND TOKENIZERS (Cached)
# ============================================================================

@st.cache_resource
def load_model():
    """Load Transformer model."""
    custom_objects = {
        'Transformer': Transformer,
        'PositionalEncoding': PositionalEncoding,
        'TransformerEncoderLayer': TransformerEncoderLayer,
        'TransformerDecoderLayer': TransformerDecoderLayer
    }
    
    # Check models folder first, then root
    model_paths = [
        'models/transformer_model.keras',
        'models/transformer_best.keras',
        'transformer_model.keras',
        'transformer_best.keras'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            return model
    
    raise FileNotFoundError("Model file not found! Please add transformer_model.keras to the models/ folder.")

@st.cache_resource
def load_tokenizers():
    """Load BPE tokenizers."""
    # Check models folder first, then root
    en_paths = ['models/en_tokenizer.model', 'en_tokenizer.model']
    hi_paths = ['models/hi_tokenizer.model', 'hi_tokenizer.model']
    
    en_path = next((p for p in en_paths if os.path.exists(p)), None)
    hi_path = next((p for p in hi_paths if os.path.exists(p)), None)
    
    if not en_path or not hi_path:
        raise FileNotFoundError("Tokenizer files not found! Please add en_tokenizer.model and hi_tokenizer.model to the models/ folder.")
    
    en_tok = BPETokenizer(en_path)
    hi_tok = BPETokenizer(hi_path)
    return en_tok, hi_tok

# ============================================================================
# 5. TRANSLATION FUNCTION
# ============================================================================

def translate(text, model, en_tokenizer, hi_tokenizer, nlp, max_len=50):
    """
    Full translation pipeline:
    1. Clean text
    2. Reorder (SVO → SOV)
    3. Encode
    4. Translate
    5. Decode
    """
    # Step 1: Clean
    text_clean = text.lower().strip()
    
    # Step 2: Reorder
    text_reordered = reorder_sentence(text_clean, nlp)
    
    # Step 3: Encode
    encoder_input = en_tokenizer.encode(text_reordered, max_len)
    
    # Step 4: Translate (greedy decoding)
    decoder_input = np.array([[hi_tokenizer.bos_id]])
    
    for _ in range(max_len):
        predictions = model.predict([encoder_input, decoder_input], verbose=0)
        next_token = np.argmax(predictions[0, -1, :])
        
        if next_token == hi_tokenizer.eos_id:
            break
        
        decoder_input = np.concatenate([decoder_input, [[next_token]]], axis=1)
    
    # Step 5: Decode
    output_ids = decoder_input[0, 1:].tolist()
    translation = hi_tokenizer.decode(output_ids)
    
    return translation, text_reordered

# ============================================================================
# 6. STREAMLIT UI
# ============================================================================

def main():
    # Header
    st.title("🇮🇳 English → Hindi Translator")
    st.markdown("**Neural Machine Translation using Transformer**")
    st.markdown("---")
    
    # Load resources
    with st.spinner("Loading model and resources..."):
        try:
            model = load_model()
            en_tokenizer, hi_tokenizer = load_tokenizers()
            nlp = load_stanza()
            model_loaded = True
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.info("Please make sure you have added the following files to the `models/` folder:\n"
                   "- `transformer_model.keras`\n"
                   "- `en_tokenizer.model`\n"
                   "- `hi_tokenizer.model`")
            model_loaded = False
    
    if model_loaded:
        st.success("✅ Model loaded successfully!")
        
        # Input
        st.markdown("### Enter English Text")
        input_text = st.text_area(
            "Type or paste English text here:",
            height=100,
            placeholder="Example: The boy eats an apple."
        )
        
        # Translate button
        translate_btn = st.button("🔄 Translate", type="primary", use_container_width=True)
        
        # Example sentences
        st.markdown("#### 💡 Try these examples:")
        
        examples = [
            "The boy eats an apple.",
            "India is a beautiful country.",
            "I love my family.",
            "She reads a book every day.",
            "The train arrives at the station."
        ]
        
        cols = st.columns(3)
        for i, ex in enumerate(examples):
            with cols[i % 3]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    input_text = ex
                    translate_btn = True
        
        # Translation
        if translate_btn and input_text.strip():
            with st.spinner("Translating..."):
                translation, reordered = translate(
                    input_text, model, en_tokenizer, hi_tokenizer, nlp
                )
            
            # Results
            st.markdown("---")
            st.markdown("### 📝 Translation Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**English (Original)**")
                st.info(input_text)
                
                if input_text.lower().strip() != reordered:
                    st.markdown("**English (Reordered → SOV)**")
                    st.warning(reordered)
            
            with col2:
                st.markdown("**हिंदी (Hindi Translation)**")
                st.success(translation)
            
            # Show preprocessing info
            st.markdown("---")
            if input_text.lower().strip() != reordered:
                st.caption("ℹ️ Source reordering (SVO → SOV) was applied to match Hindi word order.")
            else:
                st.caption("ℹ️ No reordering was needed for this sentence.")
        
        elif translate_btn:
            st.warning("⚠️ Please enter some text to translate.")
        
        # Model Info
        st.markdown("---")
        with st.expander("ℹ️ About this Model"):
            st.markdown("""
            **Architecture:** Transformer (4 layers, 256 dim, 8 heads)
            
            **Training Data:** IIT Bombay English-Hindi Parallel Corpus (310K pairs)
            
            **Performance:**
            - BLEU Score: 16.48
            - chrF Score: 38.96
            - TER Score: 74.01
            
            **Preprocessing:** 
            - BPE Tokenization (8K vocabulary)
            - Source-side reordering (SVO → SOV)
            
            **Note:** This model works best with short to medium length sentences (≤20 words).
            """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: gray; font-size: 12px;'>"
            "Built for ENPM665 NLP Course | University of Maryland"
            "</div>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
