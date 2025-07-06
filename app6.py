import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
import spacy
import numpy as np
import re
import plotly.graph_objects as go
import joblib # Import joblib for saving/loading models
import os     # Import os for path operations
import mysql.connector # Import MySQL connector for database interaction
from mysql.connector import Error

# --- Configuration for Database and Model Persistence ---
# MySQL Database Credentials
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root', # Your MySQL username (e.g., 'root')
    'password': 'passwordku123', # Your MySQL password (e.g., '')
    'database': 'hoax_detector_db' # The database name you created
}
DB_TABLE_NAME = 'news_articles' # The table name you created in MySQL

# Define paths for model persistence
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure the directory exists

MODEL_PATH = os.path.join(MODEL_DIR, 'calibrated_linearsvc_model.joblib')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
VALID_TITLES_PATH = os.path.join(MODEL_DIR, 'valid_titles.joblib')
MODEL_ACCURACY_PATH = os.path.join(MODEL_DIR, 'model_accuracy.joblib') # To save accuracy
# --- End Configuration ---

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Model spaCy 'en_core_web_sm' tidak ditemukan. Silakan install dengan: python -m spacy download en_core_web_sm")
    st.stop()

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

def extract_person_entities(text):
    """Extracts person names using spaCy NER, regex, and a hardcoded list."""
    if not text:
        return set()

    found = set()

    # 1) spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            found.add(ent.text.strip().lower())

    # 2) Regex fallback: capitalized words (min 3 chars)
    matches = re.findall(r'\b([A-Z][a-z]{2,})\b', text)
    for m in matches:
        found.add(m.lower())

    # 3) Hardcoded politician names
    for keyword in ["jokowi", "prabowo", "ganjar", "anies"]:
        if keyword in text.lower():
            found.add(keyword)

    return found

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
    
    text1_clean = text1.lower()
    text2_clean = text2.lower()
    
    # Remove all identified entities from both texts
    for entity in all_entities:
        pattern = r'\b' + re.escape(entity) + r'\b'
        text1_clean = re.sub(pattern, '', text1_clean)
        text2_clean = re.sub(pattern, '', text2_clean)
        
    # Remove punctuation and extra spaces
    text1_clean = re.sub(r'[^\w\s]', ' ', text1_clean)
    text2_clean = re.sub(r'[^\w\s]', ' ', text2_clean)
    text1_clean = ' '.join(text1_clean.split())
    text2_clean = ' '.join(text2_clean.split())

    if not text1_clean.strip() or not text2_clean.strip():
        return 0.0
    
    # Calculate TF-IDF similarity
    # Create a new vectorizer instance for each similarity calculation
    # to avoid vocabulary issues when comparing arbitrary texts
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    try:
        vectors = vectorizer.fit_transform([text1_clean, text2_clean])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except ValueError: # Handle cases where vocabulary is empty after cleaning
        return 0.0
        
    return similarity

def auto_detect_name_substitution(input_text, dataset_titles):
    """
    Detects if the input text is a name-substituted hoax by comparing it
    to valid titles in the dataset.
    """
    input_entities = extract_person_entities(input_text)
    if not input_entities:
        return False, "", set(), set(), 0.0

    # Increased threshold for more precise name substitution detection
    NAME_SUB_SIMILARITY_THRESHOLD = 0.75 

    for title in dataset_titles:
        title_entities = extract_person_entities(title)
        if not title_entities:
            continue

        content_sim = calculate_content_similarity_without_entities(input_text, title)

        # Check: similar content & different names AND high content similarity
        if content_sim >= NAME_SUB_SIMILARITY_THRESHOLD:
            # Ensure there's a difference in names, not just one missing
            if input_entities and title_entities and input_entities.symmetric_difference(title_entities):
                return True, title, input_entities, title_entities, content_sim

    return False, "", set(), set(), 0.0

def check_for_valid_match(input_text, valid_titles):
    """
    Checks if the input text is a direct match or a close paraphrase of a known
    valid title in the dataset.
    """
    VALID_MATCH_THRESHOLD = 0.85 # High threshold for strong valid match

    for title in valid_titles:
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
        bg_color = "#fff8e1"     # Background kuning muda
    else:
        gauge_color = "#27ae60"  # Hijau untuk low risk
        bg_color = "#e8f5e8"     # Background hijau muda
    
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
                {'range': [0, 30], 'color': '#d5f4e6'},   # Hijau muda
                {'range': [30, 50], 'color': '#fff2cc'},   # Kuning muda  
                {'range': [50, 70], 'color': '#ffe0b3'},   # Orange muda
                {'range': [70, 100], 'color': '#ffcccb'}   # Merah muda
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
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
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
# Attempt to load model and related data from disk at app startup
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.vectorizer = None
    st.session_state.valid_titles = []
    st.session_state.model_accuracy = 0.0

    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and \
       os.path.exists(VALID_TITLES_PATH) and os.path.exists(MODEL_ACCURACY_PATH):
        try:
            st.session_state.model = joblib.load(MODEL_PATH)
            st.session_state.vectorizer = joblib.load(VECTORIZER_PATH)
            st.session_state.valid_titles = joblib.load(VALID_TITLES_PATH)
            st.session_state.model_accuracy = joblib.load(MODEL_ACCURACY_PATH)
            st.session_state.model_trained = True
            st.success("✅ Model dan data telah dimuat dari penyimpanan lokal!")
        except Exception as e:
            st.warning(f"⚠ Gagal memuat model/data tersimpan: {e}. Akan dilatih ulang jika dataset diunggah.")
            # Reset state if loading fails to allow retraining
            st.session_state.model_trained = False
            st.session_state.model = None
            st.session_state.vectorizer = None
            st.session_state.valid_titles = []
            st.session_state.model_accuracy = 0.0

if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'current_input' not in st.session_state:
    st.session_state.current_input = ""


# Header Section
st.markdown("""
<div class="main-header">
    <div class="main-title">🛡 AI Hoax Detector Pro</div>
    <div class="main-subtitle">
        Advanced machine learning powered hoax detection system. 
        Upload your dataset, train the AI model, and analyze news articles with precision.
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
    
    # New File Uploader for adding data to DB
    st.write("---")
    st.subheader("📤 Upload New Dataset to MySQL")
    st.info("Upload an Excel file here to add its data to your MySQL database. Existing data will be preserved unless it's a duplicate.")
    new_uploaded_file = st.file_uploader(
        "Upload Excel file to populate MySQL database (.xlsx)",
        type=["xlsx"],
        key="db_uploader",
        help="Dataset must contain: 'title', 'cleaned', 'label', and other columns as per your database table. New data will be inserted."
    )

    if new_uploaded_file:
        if st.button("💾 Add Data to MySQL", use_container_width=True):
            with st.spinner("🚀 Adding data to MySQL..."):
                try:
                    df_to_add = pd.read_excel(new_uploaded_file)
                    
                    # Ensure required columns are present for DB insertion
                    required_db_cols = ['title', 'cleaned', 'label'] # Minimal columns for model training
                    # Get all columns from your SQL table definition
                    all_db_cols = ['index', 'title', 'w_timestar', 'original', 'tags', 'author', 'url', 'cleaned', 'label', 'timestamped', 'token', 'summarized']

                    # Filter DataFrame to only include columns that exist in the database table
                    df_filtered = df_to_add[[col for col in all_db_cols if col in df_to_add.columns]]

                    # Handle NaN values for text fields by converting to empty string
                    for col in ['title', 'w_timestar', 'original', 'tags', 'author', 'url', 'cleaned', 'timestamped', 'token', 'summarized']:
                        if col in df_filtered.columns:
                            df_filtered[col] = df_filtered[col].fillna('').astype(str)
                    
                    # Handle NaN for integer label, e.g., fill with -1 or drop rows
                    if 'label' in df_filtered.columns:
                        df_filtered = df_filtered.dropna(subset=['label']) # Drop rows if label is missing
                        df_filtered['label'] = df_filtered['label'].astype(int) # Ensure label is int

                    conn = None # Initialize conn to None
                    try:
                        conn = mysql.connector.connect(**DB_CONFIG)
                        cursor = conn.cursor()
                        
                        # Prepare the INSERT statement dynamically based on available columns
                        cols_to_insert = [col for col in df_filtered.columns if col in all_db_cols]
                        
                        # Exclude 'id' if it's auto-incremented in DB and not present in DF
                        if 'id' in cols_to_insert:
                            cols_to_insert.remove('id')

                        # Construct the INSERT query
                        placeholders = ', '.join(['%s'] * len(cols_to_insert))
                        columns_str = ', '.join([f'{col}' for col in cols_to_insert]) # Use backticks for column names
                        insert_query = f"INSERT INTO {DB_TABLE_NAME} ({columns_str}) VALUES ({placeholders})"
                        
                        # Convert DataFrame rows to a list of tuples for executemany
                        data_to_insert = [tuple(row) for row in df_filtered[cols_to_insert].values]
                        
                        cursor.executemany(insert_query, data_to_insert)
                        conn.commit()
                        st.success(f"✅ {len(df_to_add)} rows successfully added to MySQL database!")
                        # Invalidate trained model if new data is added, forcing a retrain
                        st.session_state.model_trained = False
                    except Error as err:
                        st.error(f"❌ Error adding data to MySQL: {err}")
                        if conn and conn.is_connected():
                            conn.rollback() # Rollback changes if an error occurred
                    finally:
                        if conn and conn.is_connected():
                            cursor.close()
                            conn.close()

                except Exception as e:
                    st.error(f"❌ Failed to process uploaded Excel file: {str(e)}")

    st.write("---") # Separator

    # Button to train model from existing MySQL data
    if st.button("🚀 Train Model from MySQL Data", type="primary", use_container_width=True):
        with st.spinner("🤖 Memuat data dari MySQL dan melatih model..."):
            try:
                # Connect to MySQL
                conn = mysql.connector.connect(**DB_CONFIG)
                cursor = conn.cursor(dictionary=True)
                
                # Fetch data - only columns needed for ML
                query = f"SELECT cleaned, label, title FROM {DB_TABLE_NAME} WHERE cleaned IS NOT NULL AND label IS NOT NULL AND title IS NOT NULL AND cleaned != '' AND title != ''"
                cursor.execute(query)
                data = cursor.fetchall()
                
                df = pd.DataFrame(data)
                
                # Close connection
                cursor.close()
                conn.close()

                if df.empty:
                    st.error("❌ Dataset yang dimuat dari MySQL kosong atau tidak memiliki kolom yang diperlukan.")
                    st.stop()
                
                # Data processing
                df_valid = df[df['label'] == 0]
                df_hoax = df[df['label'] == 1]
                
                # Balance dataset
                if len(df_hoax) < len(df_valid) * 0.3: # Adjust ratio as needed
                    n_samples = max(len(df_valid), len(df_hoax) * 3) # At least as many as valid, or 3x hoax
                    df_hoax_upsampled = resample(df_hoax, replace=True, n_samples=n_samples, random_state=42)
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
                
                acc = model.score(X_test_vec, y_test) # Use model.score for accuracy
                
                # Store in session
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                st.session_state.valid_titles = df_valid['title'].tolist() # Store valid titles
                st.session_state.model_accuracy = acc
                st.session_state.model_trained = True
                st.session_state.show_results = False
                
                # Save model and components for persistence
                joblib.dump(model, MODEL_PATH)
                joblib.dump(vectorizer, VECTORIZER_PATH)
                joblib.dump(st.session_state.valid_titles, VALID_TITLES_PATH)
                joblib.dump(acc, MODEL_ACCURACY_PATH)

                st.success(f"🎉 Model trained successfully and saved! Accuracy: {acc:.2%}")
                st.balloons()
                
            except mysql.connector.Error as err:
                st.error(f"❌ Kesalahan database: {err}. Pastikan MySQL berjalan, database '{DB_CONFIG['database']}' ada, dan tabel '{DB_TABLE_NAME}' memiliki kolom 'cleaned', 'label', 'title'.")
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

with col2:
    # Model Status Card
    if st.session_state.model_trained:
        st.markdown(f"""
        <div class="modern-card">
            <div class="card-title">🤖 Model Status</div>
            <div class="metric-card">
                <div class="metric-value">{st.session_state.model_accuracy:.1%}</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            <div style="text-align: center; margin-top: 1rem;">
                <span class="status-badge badge-success">✅ Model Ready</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="modern-card">
            <div class="card-title">🤖 Model Status</div>
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; opacity: 0.3;">🤖</div>
                <div style="color: #666; margin-top: 1rem;">No model trained yet</div>
                <span class="status-badge badge-warning">⏳ Waiting for training</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# News Analysis Section
st.markdown("""
<div class="modern-card">
    <div class="card-title">📝 News Analysis</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.model_trained:
    input_text = st.text_area(
        "Enter news text to analyze:",
        value=st.session_state.current_input,
        height=150,
        placeholder="Paste the news article or headline you want to verify..."
    )
    
    if st.button("🔍 Analyze News", type="primary", use_container_width=True):
        if input_text.strip():
            st.session_state.current_input = input_text
            st.session_state.show_results = True
        else:
            st.warning("⚠ Please enter text to analyze")
else:
    st.info("📋 Please train the model first to enable news analysis")

# Results Section
if st.session_state.model_trained and st.session_state.show_results:
    input_text = st.session_state.current_input
    
    st.markdown("""
    <div class="modern-card">
        <div class="card-title">📊 Analysis Results</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🔍 Analyzing content..."):
        # 1. Check for Name Substitution (Hoax)
        name_substituted, similar_hoax_origin_title, input_names, original_names, content_sim_hoax = \
            auto_detect_name_substitution(input_text, st.session_state.valid_titles)
        
        # 2. Check for Direct Match/Paraphrase with Valid News
        is_valid_match, matched_valid_title, content_sim_valid = \
            check_for_valid_match(input_text, st.session_state.valid_titles)

        prediction_made = False

        if is_valid_match:
            st.markdown(f"""
            <div class="result-card valid-result">
                <h2>✅ LIKELY VALID - Verified Match</h2>
                <h3>Matches a known valid news article!</h3>
                <p>This content is highly similar to a legitimate news article in our database.</p>
            </div>
            """, unsafe_allow_html=True)
            st.info(f"Matched with original valid title: \"{matched_valid_title}\" (Similarity: {content_sim_valid:.1%})")
            prediction_made = True
            
        elif name_substituted:
            st.markdown("""
            <div class="result-card hoax-result">
                <h2>🚨 HOAX ALERT - Name Substitution Detected</h2>
                <h3>This content appears to be manipulated with substituted names</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                st.markdown(f"""
                <div class="comparison-box">
                    <h4>📝 Your Input (Suspicious)</h4>
                    <p style="font-style: italic;">"{input_text}"</p>
                    <div><strong>Detected Names:</strong> {', '.join(input_names).title()}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_comp2:
                st.markdown(f"""
                <div class="comparison-box">
                    <h4>✅ Original Valid Title</h4>
                    <p style="font-style: italic;">"{similar_hoax_origin_title}"</p>
                    <div><strong>Original Names:</strong> {', '.join(original_names).title()}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.info(f"Content Similarity (excluding names): {content_sim_hoax:.1%}")
            prediction_made = True

        if not prediction_made:
            # Regular ML prediction if no specific pattern is found
            input_vector = st.session_state.vectorizer.transform([input_text])
            probas = st.session_state.model.predict_proba(input_vector)[0]
            prob_valid, prob_hoax = probas[0], probas[1]
            
            # Display result
            if prob_hoax > 0.7:
                st.markdown(f"""
                <div class="result-card hoax-result">
                    <h2>🔥 HIGH RISK HOAX</h2>
                    <div class="metric-value">{prob_hoax:.1%}</div>
                    <p>This content has a high probability of being false or misleading based on AI analysis.</p>
                </div>
                """, unsafe_allow_html=True)
            elif prob_hoax > 0.5:
                st.markdown(f"""
                <div class="result-card suspicious-result">
                    <h2>⚠ SUSPICIOUS CONTENT</h2>
                    <div class="metric-value">{prob_hoax:.1%}</div>
                    <p>This content shows signs of being potentially misleading based on AI analysis.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card valid-result">
                    <h2>✅ LIKELY VALID</h2>
                    <div class="metric-value">{prob_valid:.1%}</div>
                    <p>This content appears to be credible based on our AI analysis.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Charts - Hanya gauge di tengah
            col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
            
            with col_chart:
                fig_gauge = create_probability_gauge(prob_hoax)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
        # Common checks regardless of the primary detection
        if contains_trigger_word(input_text):
            found_triggers = [w for w in trigger_words if w.lower() in input_text.lower()]
            st.warning(f"🚨 *Warning:* Provocative language detected: {', '.join(found_triggers)}")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 4rem; padding: 2rem; background: #f8f9fa; border-radius: 15px;">
    <h4 style="color: #666;">⚠ Important Disclaimer</h4>
    <p style="color: #666; max-width: 800px; margin: 0 auto; line-height: 1.6;">
        This AI Hoax Detector is an analytical tool designed to assist in identifying potential misinformation. 
        Results should be used as guidance only. Always verify information through multiple credible sources 
        and exercise critical thinking when consuming news content.
    </p>
</div>
""", unsafe_allow_html=True)