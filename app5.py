import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
import numpy as np
import re
import plotly.graph_objects as go

# Kata-kata indikasi hoaks
trigger_words = [
    "bocor", "menggemparkan", "skandal", "konspirasi", "terbongkar",
    "geger", "heboh", "dikecam", "mencengangkan", "terungkap",
    "rahasia", "terlarang", "dilarang", "dirahasiakan", "tersembunyi"
]

def contains_trigger_word(text):
    """Checks if the text contains any of the predefined trigger words."""
    if not text:
        return False
    return any(word.lower() in text.lower() for word in trigger_words)

# --- FUNGSI PREPROCESSING UNTUK LOOKUP DAN SIMILARITAS ---
def preprocess_text_for_comparison(text):
    """
    Cleans text for consistent comparison and lookup.
    Converts to lowercase, removes punctuation, and normalizes spaces.
    This should ideally mirror how your 'cleaned' column in the dataset was prepared.
    """
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace (multiple spaces to single space)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_person_entities(text):
    """
    Extracts person names using aggressive regex and an expanded hardcoded list.
    This version does NOT use spaCy.
    """
    if not text:
        return set()

    found_names = set()
    text_lower = text.lower()

    # Hardcoded common Indonesian politician/public figure names (expanded)
    known_names = [
        "jokowi", "prabowo", "ganjar", "anies", "megawati", "sby", "ahok",
        "joko widodo", "prabowo subianto", "ganjar pranowo",
        "anies baswedan", "megawati soekarnoputri", "susilo bambang yudhoyono",
        "basuki tjahaja purnama", "ridwan kamil", "sandiaga uno", "ahy", "gus ipul",
        "ma'ruf amin", "mahfud md", "erick thohir", "budi gunadi sadikin",
        "sri mulyani", "luhut b. pandjaitan", "moeldoko", "puan maharani",
        "surya paloh", "aburizal bakrie", "agus harimurti yudhoyono"
        # Tambahkan lebih banyak nama jika diperlukan, termasuk variasi nama panggilan
    ]

    # Add known names found in the text
    for name in known_names:
        if re.search(r'\b' + re.escape(name) + r'\b', text_lower):
            found_names.add(name)

    # Regex for capitalized words or phrases (potential names)
    potential_names_regex = re.findall(r'\b([A-Z][a-z]{1,}(?:\s[A-Z][a-z]{1,})*)\b', text)
    for match in potential_names_regex:
        # Filter out common non-name capitalized words if possible (optional, but helps reduce noise)
        if match.lower() not in ["senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu",
                                 "januari", "februari", "maret", "april", "mei", "juni",
                                 "juli", "agustus", "september", "oktober", "november", "desember",
                                 "dpr", "mpr", "pki", "nkri", "ri", "pbb", "fpi", "tni", "polri", # common acronyms
                                 "indonesia", "jakarta", "bandung", "surabaya", "medan", "makassar", # major cities/countries
                                 "partai", "koalisi", "rakyat", "pemerintah", "negara", "badan", "komisi", "dewan" # common nouns
                                 ]:
            found_names.add(match.lower())
    
    # Basic rule-based extraction for names following titles
    title_patterns = {
        r"presiden\s+([a-z]+\s+[a-z]+)": 2,  # e.g., "joko widodo"
        r"presiden\s+([a-z]+)": 1,         # e.g., "jokowi"
        r"gubernur\s+([a-z]+\s+[a-z]+)": 2,
        r"gubernur\s+([a-z]+)": 1,
        r"menteri\s+([a-z]+\s+[a-z]+)": 2,
        r"menteri\s+([a-z]+)": 1,
        r"kapolri\s+([a-z]+\s+[a-z]+)": 2,
        r"kapolri\s+([a-z]+)": 1,
        r"panglima\s+tni\s+([a-z]+\s+[a-z]+)": 2,
        r"panglima\s+tni\s+([a-z]+)": 1,
        r"bapak\s+([a-z]+\s+[a-z]+)": 2,
        r"bapak\s+([a-z]+)": 1,
        r"ibu\s+([a-z]+\s+[a-z]+)": 2,
        r"ibu\s+([a-z]+)": 1,
    }

    for pattern, group_num in title_patterns.items():
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple): # If regex group captures multiple parts
                found_names.add(" ".join(match))
            else:
                found_names.add(match)

    return found_names

def calculate_content_similarity_without_entities(text1, text2):
    """
    Calculates content similarity after removing all identified person names.
    Uses TF-IDF with ngram_range(1,2).
    """
    if not text1 or not text2:
        return 0.0
    
    # Extract entities and combine for comprehensive removal
    entities1 = extract_person_entities(text1)
    entities2 = extract_person_entities(text2)
    all_entities = entities1.union(entities2)
    
    text1_clean_for_sim = preprocess_text_for_comparison(text1) # Apply consistent cleaning
    text2_clean_for_sim = preprocess_text_for_comparison(text2) # Apply consistent cleaning
    
    # Remove all identified entities from both texts
    for entity in all_entities:
        pattern = r'\b' + re.escape(entity) + r'\b'
        text1_clean_for_sim = re.sub(pattern, '', text1_clean_for_sim)
        text2_clean_for_sim = re.sub(pattern, '', text2_clean_for_sim)
        
    if not text1_clean_for_sim.strip() or not text2_clean_for_sim.strip():
        return 0.0
    
    # Calculate TF-IDF similarity
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    try:
        vectors = vectorizer.fit_transform([text1_clean_for_sim, text2_clean_for_sim])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except ValueError: # Handle cases where vocabulary is empty after cleaning
        return 0.0
        
    return similarity

def improved_detect_hoax_by_entity_difference(input_text, valid_dataset_titles, similarity_threshold=0.7):
    """
    Versi yang diperbaiki untuk deteksi hoax berdasarkan perbedaan entitas
    Hanya membandingkan dengan artikel yang memiliki kemiripan konten tinggi dari dataset valid
    """
    input_entities = extract_person_entities(input_text)
    if not input_entities:
        return False, "", set(), set()

    for title in valid_dataset_titles: # This list comes from the 'title' column of valid entries
        title_entities = extract_person_entities(title)
        if not title_entities:
            continue
        
        # Hanya cek perbedaan entitas jika konten cukup mirip
        content_similarity = calculate_content_similarity_without_entities(input_text, title)
        
        if content_similarity >= similarity_threshold:  # Konten mirip
            if input_entities.symmetric_difference(title_entities):  # Tapi entitas beda
                return True, title, input_entities, title_entities
            
    return False, "", set(), set()

def check_for_valid_match_in_dataset(input_text, valid_dataset_titles):
    """
    Checks if the input text is a direct match or a close paraphrase of a known
    valid title in the dataset.
    """
    VALID_MATCH_THRESHOLD = 0.85 # High threshold for strong valid match

    for title in valid_dataset_titles: # This list comes from the 'title' column of valid entries
        similarity = calculate_content_similarity_without_entities(input_text, title)
        if similarity >= VALID_MATCH_THRESHOLD:
            return True, title, similarity
    return False, "", 0.0
    
def create_probability_gauge(prob_hoax):
    """Membuat gauge chart modern untuk probabilitas hoax"""
    if prob_hoax >= 0.7:
        gauge_color = "#e74c3c"  # Merah terang untuk high risk
        bg_color = "#ffebee"    # Background merah muda
    elif prob_hoax >= 0.5:
        gauge_color = "#f39c12"  # Orange untuk medium risk     
        bg_color = "#fff8e1"    # Background kuning muda
    else:
        gauge_color = "#27ae60"  # Hijau untuk low risk
        bg_color = "#e8f5e8"    # Background hijau muda
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob_hoax * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': "Hoax Risk Level",  
            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Inter'}
        },
        number = {
            'font': {'size': 36, 'color': gauge_color, 'family': 'Inter'},
            'suffix': '%'
        },
        gauge = {
            'axis': {
                'range': [None, 100],  
                'tickwidth': 2,  
                'tickcolor': "#34495e",
                'tickfont': {'size': 12, 'color': '#34495e'}
            },
            'bar': {'color': gauge_color, 'thickness': 0.3},
            'bgcolor': bg_color,
            'borderwidth': 3,
            'bordercolor': gauge_color,
            'steps': [
                {'range': [0, 30], 'color': '#d5f4e6'},    # Hijau muda
                {'range': [30, 50], 'color': '#fff2cc'},    # Kuning muda  
                {'range': [50, 70], 'color': '#ffe0b3'},    # Orange muda
                {'range': [70, 100], 'color': '#ffcccb'}    # Merah muda
            ],
            'threshold': {
                'line': {'color': "#2c3e50", 'width': 3},
                'thickness': 0.8,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        font={'family': 'Inter'}
    )
    
    return fig


# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="AI Hoax Detector Pro",  
    page_icon="🛡",  
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        padding: 0;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
/* Header Styling */
.main-header {
    background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
    background-size: 400% 400%; /* Needed for the animation to work smoothly */
    animation: gradientBG 8s ease infinite; /* Apply the animation */
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 0 -2rem 2rem -2rem; /* Adjusted top margin from -2rem to 0 */
    color: white;
    border-radius: 0 0 20px 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    position: relative; /* Needed for pseudo-elements */
    overflow: hidden; /* Ensures the background animation doesn't bleed out */
}

/* Keyframes for the gradient background animation */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Optional: Add some subtle floating elements for extra visual appeal */
.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.1) 0%, transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(255,255,255,0.1) 0%, transparent 50%);
    animation: float 6s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

/* Adjust title and subtitle for better appearance within the animated header */
.main-title {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    position: relative; /* Ensure text is above pseudo-elements */
    z-index: 2; /* Ensure text is above pseudo-elements */
    animation: titleGlow 3s ease-in-out infinite alternate; /* Subtle glow for the title */
}

@keyframes titleGlow {
    from { text-shadow: 2px 2px 4px rgba(0,0,0,0.3), 0 0 20px rgba(255,255,255,0.3); }
    to { text-shadow: 2px 2px 4px rgba(0,0,0,0.3), 0 0 30px rgba(255,255,255,0.5); }
}

.main-subtitle {
    font-size: 1.2rem;
    font-weight: 300;
    opacity: 0.95;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
    position: relative; /* Ensure text is above pseudo-elements */
    z-index: 2; /* Ensure text is above pseudo-elements */
    animation: subtitleFade 2s ease-in-out; /* Fade-in effect for subtitle */
}

@keyframes subtitleFade {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 0.95; transform: translateY(0); }
}
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1rem;
        font-weight: 300;
        opacity: 0.9;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.4;
    }
    
    /* Card Styling */
    .modern-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.12);
    }
    
    .card-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box_shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Input Styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File Uploader */
    .stFileUploader > div {
        border-radius: 12px;
        border: 2px dashed #cbd5e0;
        padding: 2rem;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea;
        background-color: #f7fafc;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Alert Styling */
    .stAlert {
        border-radius: 15px;
        border: none;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .stError {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #8b4513;
    }
    
    /* Result Cards */
    .result-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .hoax-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }
    
    .valid-result {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
    }
    
    .suspicious-result {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Comparison styling */
    .comparison-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .badge-success { background: #d4edda; color: #155724; }
    .badge-danger { background: #f8d7da; color: #721c24; }
    .badge-warning { background: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'current_input' not in st.session_state:
    st.session_state.current_input = ""
if 'valid_titles' not in st.session_state:
    st.session_state.valid_titles = []
if 'full_dataset_lookup' not in st.session_state:
    st.session_state.full_dataset_lookup = {} # To store cleaned text -> label for direct match
if 'original_titles_and_labels' not in st.session_state:
    st.session_state.original_titles_and_labels = [] # New: To store (original_title, label) pairs

# Header Section
st.markdown("""
<div class="main-header">
    <div class="main-title">🛡 AI Hoax Detector Pro</div>
    <div class="main-subtitle">
        Sistem deteksi hoaks canggih bertenaga AI.  
        Unggah dataset Anda, latih model AI, dan analisis artikel berita dengan presisi.
    </div>
</div>
""", unsafe_allow_html=True)

# Main content in cards
col1, col2 = st.columns([2, 1])

with col1:
    # Dataset Upload Card
    st.markdown("""
    <div class="modern-card">
        <div class="card-title">📊 Dataset & Model Training</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Unggah dataset Excel Anda (.xlsx)",
        type=["xlsx"],
        help="Dataset harus mengandung kolom: 'cleaned' (teks), 'label' (0=valid, 1=hoaks), 'title'"
    )
    
    if uploaded_file:
        st.success("✅ Dataset berhasil diunggah!")
        
        if not st.session_state.model_trained:
            if st.button("🚀 Latih Model", type="primary", use_container_width=True):
                with st.spinner("🤖 Melatih model AI... Mohon tunggu..."):
                    try:
                        df = pd.read_excel(uploaded_file)
                        
                        if all(col in df.columns for col in ['cleaned', 'label', 'title']):
                            # Data processing
                            df = df.dropna(subset=['cleaned', 'label', 'title'])
                            df = df[df['cleaned'].str.strip() != '']
                            df = df[df['title'].str.strip() != '']
                            
                            df_valid = df[df['label'] == 0]
                            df_hoax = df[df['label'] == 1]
                            
                            # --- PENTING: Penyimpanan data untuk pemeriksaan langsung ---
                            # Menggunakan kolom 'cleaned' sebagai kunci untuk lookup label
                            st.session_state.full_dataset_lookup = df.set_index('cleaned')['label'].to_dict()
                            
                            # Menyimpan judul asli dan labelnya untuk pencocokan prioritas tertinggi
                            st.session_state.original_titles_and_labels = df[['title', 'label']].values.tolist()
                            # --- Akhir bagian penting ---

                            # Balance dataset (oversampling hoax class if it's too small)
                            if len(df_hoax) < len(df_valid) * 0.3: # If hoax is less than 30% of valid
                                n_samples_hoax = max(len(df_valid), len(df_hoax) * 3) # At least as many as valid, or 3x hoax
                                df_hoax_upsampled = resample(df_hoax, replace=True, n_samples=n_samples_hoax, random_state=42)
                                df_balanced = pd.concat([df_valid, df_hoax_upsampled])
                            else:
                                df_balanced = df.copy()
                            
                            # Train model
                            X = df_balanced['cleaned']
                            y = df_balanced['label']
                            
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
                            
                            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_df=0.8, min_df=3, max_features=5000)
                            X_train_vec = vectorizer.fit_transform(X_train)
                            X_test_vec = vectorizer.transform(X_test)
                            
                            base_model = LinearSVC(C=0.1, max_iter=2000, random_state=42)
                            model = CalibratedClassifierCV(estimator=base_model, cv=5)
                            model.fit(X_train_vec, y_train)
                            
                            # Store in session
                            st.session_state.model = model
                            st.session_state.vectorizer = vectorizer
                            st.session_state.valid_titles = df_valid['title'].tolist() # Mengambil kolom 'title' dari data yang berlabel 0 (valid)
                            st.session_state.model_accuracy = 0.0 # Placeholder, as full pipeline accuracy is complex to measure here without spaCy
                            st.session_state.model_trained = True
                            st.session_state.show_results = False # Reset show_results after training
                            
                            st.success("🎉 Model berhasil dilatih! Akurasi Model akan diukur saat analisis.")
                            st.balloons()
                            
                        else:
                            st.error("❌ Format dataset tidak valid. Pastikan nama kolom benar ('cleaned', 'label', 'title').")
                            
                    except Exception as e:
                        st.error(f"❌ Pelatihan gagal: {str(e)}")

with col2:
    # Model Status Card
    if st.session_state.model_trained:
        st.markdown(f"""
        <div class="modern-card">
            <div class="card-title">🤖 Status Model</div>
            <div class="metric-card">
                <div class="metric-value">N/A</div>
                <div class="metric-label">Akurasi Model Pipeline</div>
            </div>
            <div style="text-align: center; margin-top: 1rem;">
                <span class="status-badge badge-success">✅ Model Siap</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="modern-card">
            <div class="card-title">🤖 Status Model</div>
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; opacity: 0.3;">🤖</div>
                <div style="color: #666; margin-top: 1rem;">Belum ada model yang dilatih</div>
                <span class="status-badge badge-warning">⏳ Menunggu pelatihan</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# News Analysis Section
st.markdown("""
<div class="modern-card">
    <div class="card-title">📝 Analisis Berita</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.model_trained:
    input_text = st.text_area(
        "Masukkan teks berita untuk dianalisis:",
        value=st.session_state.current_input,
        height=150,
        placeholder="Tempel artikel berita atau judul yang ingin Anda verifikasi di sini..."
    )
    
    if st.button("🔍 Analisis Berita", type="primary", use_container_width=True):
        if input_text.strip():
            st.session_state.current_input = input_text
            st.session_state.show_results = True
        else:
            st.warning("⚠ Mohon masukkan teks untuk dianalisis")
else:
    st.info("📋 Mohon latih model terlebih dahulu untuk mengaktifkan analisis berita")

# Results Section
if st.session_state.model_trained and st.session_state.show_results:
    input_text = st.session_state.current_input
    
    st.markdown("""
    <div class="modern-card">
        <div class="card-title">📊 Hasil Analisis</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🔍 Menganalisis konten..."):
        # Initial prediction variables
        final_prediction_status = None 
        prob_hoax = 0.0 

        # Always calculate ML probabilities, as they are used for the gauge regardless of final status
        input_vector = st.session_state.vectorizer.transform([input_text])
        probas = st.session_state.model.predict_proba(input_vector)[0]
        prob_hoax = probas[1] # Probability for the 'hoax' class (label 1)

        # --- REVISED PRIORITAS PENENTUAN STATUS (FINAL FINAL VERSION) ---
        
        # Flag untuk menandai apakah status sudah ditemukan oleh aturan prioritas tinggi
        found_final_status = False 

        # Pre-calculate all heuristic flags and direct match labels once
        is_input_provocative = contains_trigger_word(input_text)
        is_hoax_entity_diff, original_title_for_hoax, input_names, original_names = \
            improved_detect_hoax_by_entity_difference(input_text, st.session_state.valid_titles)
        is_valid_paraphrase_match, matched_valid_title, content_sim_valid = \
            check_for_valid_match_in_dataset(input_text, st.session_state.valid_titles)

        input_text_processed_for_lookup = preprocess_text_for_comparison(input_text)
        direct_match_label_cleaned = st.session_state.full_dataset_lookup.get(input_text_processed_for_lookup, None)

        direct_original_title_label = None
        MATCH_TITLE_THRESHOLD = 0.98 
        for original_title_from_dataset, original_label_from_dataset in st.session_state.original_titles_and_labels:
            title_similarity = calculate_content_similarity_without_entities(input_text, original_title_from_dataset)
            if title_similarity >= MATCH_TITLE_THRESHOLD:
                direct_original_title_label = original_label_from_dataset
                break # Found a very close title match, break loop


        # Fase 1: Deteksi HOAX Definitif (Prioritas Tertinggi)
        # Kondisi HOAX yang paling kuat akan diperiksa pertama

        # 1a. Direct Original Title Match - HOAX
        if direct_original_title_label == 1:
            final_prediction_status = 'direct_original_title_hoax_match'
            found_final_status = True
        
        # 1b. Direct Cleaned Text Match - HOAX (hanya jika status belum ditemukan)
        if not found_final_status:
            if direct_match_label_cleaned == 1:
                final_prediction_status = 'direct_hoax_match'
                found_final_status = True
        
        # 1c. Deteksi Perbedaan Entitas (hanya jika status belum ditemukan)
        if not found_final_status:
            if is_hoax_entity_diff:
                final_prediction_status = 'hoax_entity_diff'
                found_final_status = True
        
        # 1d. Prediksi Model ML: HOAX (jika status belum ditemukan) - Ini adalah KUNCI perubahan!
        # Memastikan bahwa jika SVM mendeteksi hoax, ia tidak akan ditimpa oleh "provokatif"
        if not found_final_status:
            if prob_hoax > 0.7: # SVM confidently predicts hoax
                final_prediction_status = 'hoax' 
                found_final_status = True
        
        # Fase 2: Deteksi VALID atau WASPADA (Jika BELUM ada HOAX Definitif dari Fase 1)
        if not found_final_status:
            # 2a. Direct Original Title Match - VALID (dengan/tanpa kata provokatif)
            if direct_original_title_label == 0:
                if is_input_provocative: # Valid dari dataset, tapi ada provokatif
                    final_prediction_status = 'direct_original_title_valid_with_triggers'
                else:
                    final_prediction_status = 'direct_original_title_valid_match'
                found_final_status = True
            
            # 2b. Direct Cleaned Text Match - VALID (dengan/tanpa kata provokatif)
            if not found_final_status:
                if direct_match_label_cleaned == 0:
                    if is_input_provocative: # Valid dari dataset, tapi ada provokatif
                        final_prediction_status = 'direct_valid_with_triggers' 
                    else:
                        final_prediction_status = 'direct_valid_match'
                    found_final_status = True
            
            # 2c. Pencocokan Parafrase Valid (dengan/tanpa kata provokatif)
            if not found_final_status:
                if is_valid_paraphrase_match:
                    if is_input_provocative: # Parafrase valid, tapi ada provokatif
                        # Berita valid dengan kata provokatif: WASPADA. Ini adalah hasil yang diinginkan.
                        final_prediction_status = 'suspicious_with_triggers' 
                    else:
                        final_prediction_status = 'valid_verified'
                    found_final_status = True
            
            # 2d. Deteksi Kata Provokatif Umum (hanya jika tidak ada aturan lain yang terpicu)
            # Ini adalah skenario di mana konten tidak terklasifikasi kuat sebagai hoax/valid,
            # namun memiliki kata provokatif.
            if not found_final_status:
                if is_input_provocative:
                    final_prediction_status = 'suspicious_with_triggers' 
                    found_final_status = True
                
                # 2e. Prediksi Model ML (Fallback): Suspicious / Valid
                # Ini adalah fallback terakhir jika tidak ada aturan heuristik yang terpicu.
                # Perhatikan bahwa kasus prob_hoax > 0.7 sudah ditangani di 1d.
                if not found_final_status: 
                    if prob_hoax > 0.5: # SVM predicts suspicious
                        final_prediction_status = 'suspicious'
                    else: # SVM predicts valid
                        final_prediction_status = 'valid'

    # --- MENAMPILKAN HASIL BERDASARKAN final_prediction_status ---
    
    # NEW: Direct Original Title Match (Hoax)
    if final_prediction_status == 'direct_original_title_hoax_match':
        st.markdown(f"""
        <div class="result-card hoax-result">
            <h2>🚨 HOAKS TERDETEKSI! (Pencocokan Judul Asli Dataset)</h2>
            <h3>Teks ini sangat mirip dengan judul berita yang **berlabel hoaks** di database asli Anda.</h3>
            <div class="metric-value">{prob_hoax:.1%}</div>
            <p>Probabilitas hoaks berdasarkan model AI.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("Peringatan: Konten ini telah diverifikasi sebagai hoaks dalam dataset Anda berdasarkan judul aslinya.")
        col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
        with col_chart:
            fig_gauge = create_probability_gauge(prob_hoax)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # NEW: Direct Original Title Match (Valid)
    elif final_prediction_status == 'direct_original_title_valid_match':
        st.markdown(f"""
        <div class="result-card valid-result">
            <h2>✅ VALID (Pencocokan Judul Asli Dataset)</h2>
            <h3>Teks ini sangat mirip dengan judul berita yang **berlabel valid** di database asli Anda.</h3>
            <div class="metric-value">{(1 - prob_hoax):.1%}</div>
            <p>Probabilitas valid berdasarkan model AI.</p>
        </div>
        """, unsafe_allow_html=True)
        st.success("Konten ini telah diverifikasi valid dalam dataset Anda berdasarkan judul aslinya.")
        col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
        with col_chart:
            fig_gauge = create_probability_gauge(prob_hoax)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # NEW: Direct Original Title Match (Valid) but with Triggers
    elif final_prediction_status == 'direct_original_title_valid_with_triggers':
        st.markdown(f"""
        <div class="result-card suspicious-result">
            <h2>⚠ KONTEN MENCURIGAKAN (Judul Dataset Valid - Ada Kata Provokatif)</h2>
            <h3>Teks ini sangat mirip dengan judul berita yang **berlabel valid** di database asli Anda, NAMUN terdeteksi bahasa provokatif.</h3>
            <div class="metric-value">{prob_hoax:.1%}</div>
            <p>Probabilitas hoaks berdasarkan model AI.</p>
        </div>
        """, unsafe_allow_html=True)
        st.warning("Peringatan: Meskipun konten ini diverifikasi valid di dataset, kata-kata provokatif terdeteksi. Harap waspada!")
        col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
        with col_chart:
            fig_gauge = create_probability_gauge(prob_hoax)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Direct Match (Hoax - from 'cleaned' column lookup)
    elif final_prediction_status == 'direct_hoax_match':
        st.markdown(f"""
        <div class="result-card hoax-result">
            <h2>🚨 HOAKS TERDETEKSI! (Pencocokan Teks Bersih Dataset)</h2>
            <h3>Konten ini cocok persis dengan teks bersih yang **berlabel hoaks** di database pelatihan.</h3>
            <div class="metric-value">{prob_hoax:.1%}</div>
            <p>Probabilitas hoaks berdasarkan model AI.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("Peringatan: Konten ini telah diverifikasi sebagai hoaks dalam dataset Anda berdasarkan teks yang sudah dibersihkan.")
        col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
        with col_chart:
            fig_gauge = create_probability_gauge(prob_hoax)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Direct Match (Valid - from 'cleaned' column lookup)
    elif final_prediction_status == 'direct_valid_match':
        st.markdown(f"""
        <div class="result-card valid-result">
            <h2>✅ VALID (Pencocokan Teks Bersih Dataset)</h2>
            <h3>Konten ini cocok persis dengan teks bersih yang **berlabel valid** di database pelatihan.</h3>
            <div class="metric-value">{(1 - prob_hoax):.1%}</div>
            <p>Probabilitas valid berdasarkan model AI.</p>
        </div>
        """, unsafe_allow_html=True)
        st.success("Konten ini telah diverifikasi valid dalam dataset Anda berdasarkan teks yang sudah dibersihkan.")
        col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
        with col_chart:
            fig_gauge = create_probability_gauge(prob_hoax)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Direct Match (Valid - from 'cleaned' column lookup) but with Triggers
    elif final_prediction_status == 'direct_valid_with_triggers':
        st.markdown(f"""
        <div class="result-card suspicious-result">
            <h2>⚠ KONTEN MENCURIGAKAN (Teks Bersih Dataset Valid - Ada Kata Provokatif)</h2>
            <h3>Konten ini cocok persis dengan teks bersih yang **berlabel valid** di database pelatihan, NAMUN terdeteksi bahasa provokatif.</h3>
            <div class="metric-value">{prob_hoax:.1%}</div>
            <p>Probabilitas hoaks berdasarkan model AI.</p>
        </div>
        """, unsafe_allow_html=True)
        st.warning("Peringatan: Meskipun konten ini diverifikasi valid di dataset, kata-kata provokatif terdeteksi. Harap waspada!")
        col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
        with col_chart:
            fig_gauge = create_probability_gauge(prob_hoax)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Suspicious because of Trigger Words (new higher priority)
    elif final_prediction_status == 'suspicious_with_triggers':
        st.markdown(f"""
        <div class="result-card suspicious-result">
            <h2>⚠ KONTEN MENCURIGAKAN (Kata Provokatif Terdeteksi)</h2>
            <h3>Terdapat kata-kata yang mengindikasikan narasi provokatif atau sensasional. Harap teliti lebih lanjut.</h3>
            <div class="metric-value">{prob_hoax:.1%}</div>
            <p>Probabilitas hoaks berdasarkan model AI.</p>
        </div>
        """, unsafe_allow_html=True)
        col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
        with col_chart:
            fig_gauge = create_probability_gauge(prob_hoax)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Hoax due to Entity Difference (now lower priority than direct/triggers)
    elif final_prediction_status == 'hoax_entity_diff':
        st.markdown("""
        <div class="result-card hoax-result">
            <h2>🚨 HOAKS TERDETEKSI - Perbedaan Entitas!</h2>
            <h3>Ditemukan perbedaan signifikan pada nama entitas dibandingkan berita valid yang dikenal. Ini kemungkinan besar upaya manipulasi.</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_comp1, col_comp2 = st.columns(2)
        with col_comp1:
            st.markdown(f"""
            <div class="comparison-box">
                <h4>📝 Input Anda (Mencurigakan)</h4>
                <p style="font-style: italic;">"{input_text}"</p>
                <div><strong>Nama Terdeteksi:</strong> {', '.join(input_names).title() if input_names else 'Tidak ada'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_comp2:
            st.markdown(f"""
            <div class="comparison-box">
                <h4>✅ Judul Berita Valid yang Cocok</h4>
                <p style="font-style: italic;">"{original_title_for_hoax}"</p>
                <div><strong>Nama Asli:</strong> {', '.join(original_names).title() if original_names else 'Tidak ada'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("Perbedaan nama entitas menjadi indikator utama hoaks ini.")
        col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
        with col_chart:
            fig_gauge = create_probability_gauge(prob_hoax)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Valid because of Paraphrase Match (now lower priority)
    elif final_prediction_status == 'valid_verified':
        st.markdown(f"""
        <div class="result-card valid-result">
            <h2>✅ KEMUNGKINAN VALID - Terverifikasi Cocok</h2>
            <h3>Sangat mirip dengan artikel berita valid yang dikenal!</h3>
            <p>Konten ini memiliki kemiripan tinggi dengan artikel berita sah di database kami.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info(f"Cocok dengan judul valid asli: \"{matched_valid_title}\" (Kemiripan: {content_sim_valid:.1%})")
        col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
        with col_chart:
            fig_gauge = create_probability_gauge(prob_hoax)
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    # ML Model predicts high hoax probability (lowest priority for hoax)
    elif final_prediction_status == 'hoax':
        st.markdown(f"""
        <div class="result-card hoax-result">
            <h2>🔥 RISIKO TINGGI HOAKS</h2>
            <div class="metric-value">{prob_hoax:.1%}</div>
            <p>Konten ini memiliki probabilitas tinggi sebagai informasi palsu atau menyesatkan berdasarkan analisis AI.</p>
        </div>
        """, unsafe_allow_html=True)
        col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
        with col_chart:
            fig_gauge = create_probability_gauge(prob_hoax)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # ML Model predicts suspicious probability (lowest priority for suspicious)
    elif final_prediction_status == 'suspicious':
        st.markdown(f"""
        <div class="result-card suspicious-result">
            <h2>⚠ KONTEN MENCURIGAKAN</h2>
            <div class="metric-value">{prob_hoax:.1%}</div>
            <p>Konten ini menunjukkan tanda-tanda berpotensi menyesatkan berdasarkan analisis AI.</p>
        </div>
        """, unsafe_allow_html=True)
        col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
        with col_chart:
            fig_gauge = create_probability_gauge(prob_hoax)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # ML Model predicts low hoax probability (lowest priority for valid)
    elif final_prediction_status == 'valid':
        st.markdown(f"""
        <div class="result-card valid-result">
            <h2>✅ KEMUNGKINAN VALID</h2>
            <div class="metric-value">{(1 - prob_hoax):.1%}</div>
            <p>Konten ini tampak kredibel berdasarkan analisis AI kami.</p>
        </div>
        """, unsafe_allow_html=True)
        col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
        with col_chart:
            fig_gauge = create_probability_gauge(prob_hoax) 
            st.plotly_chart(fig_gauge, use_container_width=True)
    

# # --- Entity Extraction Debugger Section ---
# st.markdown("""
# ---
# <div class="modern-card">
#     <div class="card-title">🔬 Debugger Ekstraksi Entitas</div>
# </div>
# """, unsafe_allow_html=True)

# st.info("""
# Gunakan bagian ini untuk menguji bagaimana sistem mendeteksi nama orang (entitas) dari teks yang Anda berikan. 
# *CATATAN PENTING:* Versi ini TIDAK menggunakan model spaCy. Deteksi entitas murni bergantung pada ekspresi reguler dan daftar nama hardcoded. Akurasi mungkin bervariasi.
# """)

# debug_entity_input = st.text_area(
#     "Masukkan teks untuk menguji ekstraksi entitas:",
#     value="Anies di Milad BKMT: Pengajian Menghasilkan Ibu-ibu Berpenpengetahuan. Presiden Joko Widodo dan Prabowo Subianto juga hadir.",
#     height=100,
#     key="debug_entity_text_area"
# )

# if st.button("🔎 Deteksi Entitas", key="detect_entities_button"):
#     if debug_entity_input.strip():
#         extracted_names = extract_person_entities(debug_entity_input)
#         st.markdown(f"*Entitas (Nama Orang) Terdeteksi:*")
#         if extracted_names:
#             st.success(f"Ditemukan: {', '.join(sorted(list(extracted_names))).title()}")
#         else:
#             st.warning("Tidak ada nama orang yang terdeteksi.")
#         st.markdown(f"---")
#     else:
#         st.warning("Silakan masukkan teks untuk pengujian entitas.")

# Footer
st.markdown("""
---
<div style="text-align: center; margin-top: 4rem; padding: 2rem; background: #f8f9fa; border-radius: 15px;">
    <h4 style="color: #666;">⚠ Disclaimer Penting</h4>
    <p style="color: #666; max-width: 800px; margin: 0 auto; line-height: 1.6;">
        AI Hoax Detector ini adalah alat analisis yang dirancang untuk membantu mengidentifikasi potensi misinformasi.  
        Hasil harus digunakan sebagai panduan saja. Selalu verifikasi informasi melalui berbagai sumber kredibel  
        dan terapkan pemikiran kritis saat mengonsumsi konten berita.
    </p>
</div>
""", unsafe_allow_html=True)