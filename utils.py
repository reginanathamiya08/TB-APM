import pandas as pd
import re
import spacy
from difflib import SequenceMatcher
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

POLITICAL_PARTIES = {
    "pkb", "partai kebangkitan bangsa",
    "gerindra", "partai gerindra",
    "golkar", "partai golkar",
    "pdip", "pdi-p", "partai demokrasi indonesia perjuangan",
    "demokrat", "partai demokrat",
    "pan", "partai amanat nasional",
    "pks", "partai keadilan sejahtera",
    "nasdem", "partai nasdem",
    "ppp", "partai persatuan pembangunan",
    "hanura", "partai hati nurani rakyat",
    "psi", "partai solidaritas indonesia",
    "perindo", "partai persatuan indonesia",
    "pkpi", "partai keadilan dan persatuan indonesia",
    "berkarya", "partai berkarya"
}

BERT_MODEL = None
NLP = None
NB_MODEL_PIPELINE = None

def load_models_base():
    bert_model = SentenceTransformer("indobenchmark/indobert-base-p1")
    nlp = spacy.load("xx_ent_wiki_sm")
    return bert_model, nlp

def build_enhanced_synonym_dict_base():
    # ...isi tetap seperti sebelumnya...
    base_synonyms = {
        "bakal": "akan", "bakalan": "akan", "hendak": "akan", "mau": "akan",
        "akan": "akan", "mesti": "harus", "kudu": "harus", "wajib": "harus",
        "bilang": "kata", "ucap": "kata", "sebut": "kata", "tutur": "kata",
        "omong": "kata", "cerita": "kata", "ungkap": "kata", "sampaikan": "kata",
        "gak": "tidak", "ga": "tidak", "nggak": "tidak", "enggak": "tidak",
        "tak": "tidak", "ndak": "tidak", "nda": "tidak", "bukan": "tidak",
        "lagi": "sedang", "udah": "sudah", "udahan": "sudah", "dah": "sudah",
        "telah": "sudah", "pernah": "sudah", "sempat": "sudah",
        "esok": "besok", "kemarin": "kemarin", "tadi": "kemarin", "tempo": "lalu",
        "tapi": "tetapi", "namun": "tetapi", "akan tetapi": "tetapi", "cuma": "tetapi",
        "kecuali": "tetapi", "selain": "kecuali", "dan": "dan", "serta": "dan",
        "gimana": "bagaimana", "kenapa": "mengapa", "dimana": "di mana",
        "kayak": "seperti", "kaya": "seperti", "kayaknya": "sepertinya",
        "macam": "seperti", "mirip": "seperti", "ibarat": "seperti",
        "banget": "sangat", "sekali": "sangat", "amat": "sangat", "bener": "sangat",
        "parah": "sangat", "ekstrem": "sangat", "luar biasa": "sangat",
        "sama": "dengan", "ama": "dengan", "ma": "dengan", "bareng": "dengan",
        "bersama": "dengan", "beserta": "dengan",
        "pemda": "pemerintah daerah", "pemkot": "pemerintah kota",
        "pemkab": "pemerintah kabupaten", "presiden": "presiden",
        "menteri": "menteri", "gubernur": "gubernur", "bupati": "bupati",
        "walikota": "walikota", "camat": "camat", "lurah": "lurah",
        "satu": "1", "dua": "2", "tiga": "3", "empat": "4", "lima": "5",
        "enam": "6", "tujuh": "7", "delapan": "8", "sembilan": "9", "sepuluh": "10",
        "puluhan": "banyak", "ratusan": "banyak", "ribuan": "banyak",
        "kabar": "berita", "info": "informasi", "laporan": "berita", "warta": "berita",
        "news": "berita", "breaking": "terbaru", "update": "terbaru", "terkini": "terbaru",
        "datang": "tiba", "pergi": "berangkat", "pulang": "kembali", "balik": "kembali",
        "cabut": "pergi", "nongol": "datang", "muncul": "datang", "hadir": "datang",
        "bagus": "baik", "jelek": "buruk", "gede": "besar", "kecil": "kecil",
        "mantap": "baik", "keren": "baik", "buruk": "buruk", "parah": "buruk",
        "oke": "baik", "ok": "baik", "fine": "baik",
        "doi": "dia", "nyokap": "ibu", "bokap": "ayah", "ortu": "orang tua",
        "gue": "saya", "gw": "saya", "ane": "saya", "lu": "kamu", "lo": "kamu",
        "ente": "kamu", "bro": "saudara", "sis": "saudara",
        "ngaku": "mengaku", "ngomong": "bicara", "ngasih": "memberi",
        "ngambil": "mengambil", "ngeliat": "melihat", "ngedenger": "mendengar",
        "nggih": "ya", "injih": "ya", "iya": "ya", "yoi": "ya", "yup": "ya",
        "enggeh": "ya", "oke": "ya", "siap": "ya",
        "gadget": "perangkat", "smartphone": "ponsel", "laptop": "komputer",
        "online": "daring", "offline": "luring", "update": "pembaruan",
        "duit": "uang", "perak": "uang", "cuan": "keuntungan", "untung": "keuntungan",
        "rugi": "kerugian", "bangkrut": "pailit", "sukses": "berhasil"
    }
    return base_synonyms

# Dictionary sinonim utama (akan diupdate saat load data)
SYNONYM_DICT = {
    "gak": "tidak",
    "nggak": "tidak",
    "enggak": "tidak",
}

def normalize_synonyms(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    words = text.split()
    normalized_words = []
    i = 0
    while i < len(words):
        word = words[i]
        if i < len(words) - 1:
            two_word = f"{word} {words[i+1]}"
            if two_word in SYNONYM_DICT:
                normalized_words.append(SYNONYM_DICT[two_word])
                i += 2
                continue
        if word in SYNONYM_DICT:
            normalized_words.append(SYNONYM_DICT[word])
        else:
            normalized_words.append(word)
        i += 1
    return ' '.join(normalized_words)

def set_global_models(bert_model, nlp_model):
    global BERT_MODEL, NLP
    BERT_MODEL = bert_model
    NLP = nlp_model

def set_global_nb_pipeline(pipeline):
    global NB_MODEL_PIPELINE
    NB_MODEL_PIPELINE = pipeline

def is_meaning_paraphrase(text1, text2, similarity_threshold=0.78):
    if BERT_MODEL is None:
        raise RuntimeError("BERT_MODEL not initialized. Call set_global_models() first.")
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    emb1 = BERT_MODEL.encode(norm1, convert_to_tensor=True)
    emb2 = BERT_MODEL.encode(norm2, convert_to_tensor=True)
    semantic_sim = util.pytorch_cos_sim(emb1, emb2).item()
    matcher = SequenceMatcher(None, norm1.split(), norm2.split())
    structural_sim = matcher.ratio()
    combined_score = 0.8 * semantic_sim + 0.2 * structural_sim
    return combined_score >= similarity_threshold

def compare_entities(input_entities, matched_entities, required_match_ratio=0.8, fuzzy_threshold=80):
    if not input_entities and not matched_entities:
        return True
    if not input_entities or not matched_entities:
        return False
    matched_input_entities_count = 0
    temp_matched_entities = list(matched_entities)
    for i_ent in input_entities:
        for m_ent in temp_matched_entities:
            if fuzz.token_set_ratio(i_ent, m_ent) >= fuzzy_threshold:
                matched_input_entities_count += 1
                temp_matched_entities.remove(m_ent)
                break
    if len(input_entities) == 1 and len(matched_entities) == 1 and \
       fuzz.token_set_ratio(list(input_entities)[0], list(matched_entities)[0]) < fuzzy_threshold:
        return False
    input_match_ratio = matched_input_entities_count / len(input_entities)
    return input_match_ratio >= required_match_ratio

def normalize_text(text):
    if not text or pd.isna(text):
        return ""
    stop_factory = StopWordRemoverFactory()
    stop_words = set(stop_factory.get_stop_words())
    stop_words.discard('akan')
    stemmer = StemmerFactory().create_stemmer()
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = normalize_synonyms(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    tokens = [stemmer.stem(w) for w in tokens if len(w) > 3]
    return ' '.join(tokens)

def extract_entities(text):
    if NLP is None:
        raise RuntimeError("NLP model not initialized. Call set_global_models() first.")
    doc = NLP(text)
    entities = set()
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE']:
            entities.add(ent.text.lower())
    text_lower = text.lower()
    for party in POLITICAL_PARTIES:
        if re.search(r'\b' + re.escape(party) + r'\b', text_lower):
            entities.add(party)
    return entities

def extract_political_parties(text):
    found_parties = set()
    text_lower = text.lower()
    temp_words = text_lower.split()
    normalized_temp_words = []
    for word in temp_words:
        normalized_temp_words.append(SYNONYM_DICT.get(word, word))
    processed_text = " ".join(normalized_temp_words)
    for party_name_or_alias in POLITICAL_PARTIES:
        if re.search(r'\b' + re.escape(party_name_or_alias) + r'\b', processed_text):
            found_parties.add(party_name_or_alias)
    canonical_parties = set()
    for found_party in found_parties:
        if found_party in SYNONYM_DICT and SYNONYM_DICT[found_party] in POLITICAL_PARTIES:
            canonical_parties.add(SYNONYM_DICT[found_party])
        elif found_party in POLITICAL_PARTIES:
            canonical_parties.add(found_party)
        else:
            for canonical in POLITICAL_PARTIES:
                if fuzz.token_set_ratio(found_party, canonical) > 90:
                    canonical_parties.add(canonical)
                    break
    return canonical_parties

def summarize_text_simple(text, num_sentences=2):
    if not text or pd.isna(text):
        return "Tidak ada rangkuman."
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return "Tidak ada rangkuman."
    summary = " ".join(sentences[:num_sentences])
    if summary and summary[-1] not in ['.', '!', '?']:
        summary += "."
    return summary

def load_and_process_data_base(data_path="book1.xlsx"):
    df = pd.read_excel(data_path)
    # Gabungkan semua sinonim ke SYNONYM_DICT
    synonym_dict_generated = auto_generate_synonyms_base(df)
    SYNONYM_DICT.update(build_enhanced_synonym_dict_base())
    SYNONYM_DICT.update(synonym_dict_generated)
    df['text_norm'] = df['title'].apply(normalize_text)
    df['entities'] = df['title'].apply(lambda x: extract_entities(str(x)))
    df['political_parties'] = df['title'].apply(extract_political_parties)
    nb_pipeline = None
    if 'label' not in df.columns:
        print("Warning: 'label' column not found in data. Cannot train Naive Bayes.")
    else:
        nb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1,2))),
            ('nb', MultinomialNB())
        ])
        train_df = df.dropna(subset=['text_norm', 'label'])
        if not train_df.empty:
            train_df['label'] = train_df['label'].astype(int)
            if len(train_df['label'].unique()) < 2:
                print("Warning: Naive Bayes model requires at least two classes (0 and 1) for training. Only one class found.")
                nb_pipeline = None
            else:
                nb_pipeline.fit(train_df['text_norm'], train_df['label'])
                print("Naive Bayes model trained successfully.")
                joblib.dump(nb_pipeline, 'naive_bayes_pipeline.pkl')
        else:
            print("No valid data for training Naive Bayes model.")
            nb_pipeline = None
    set_global_nb_pipeline(nb_pipeline)
    return df, synonym_dict_generated

def auto_generate_synonyms_base(_df):
    try:
        auto_synonyms = build_enhanced_synonym_dict_base().copy()
        all_words = []
        word_pairs = []
        for title in _df['title'].dropna():
            words = str(title).lower().split()
            clean_words = [re.sub(r'[^\w]', '', w) for w in words if len(w) > 2]
            all_words.extend(clean_words)
            for i, word in enumerate(clean_words):
                for j, other_word in enumerate(clean_words):
                    if i != j and abs(i-j) <= 3:
                        word_pairs.append((word, other_word))
        dataset_patterns = {
            "dikabarkan": "diberitakan", "dilaporkan": "diberitakan",
            "diklaim": "dinyatakan", "diungkap": "dikatakan",
            "terungkap": "diketahui", "terbongkar": "diketahui",
            "mencuat": "muncul", "merebak": "menyebar",
            "viral": "terkenal", "heboh": "ramai", "gaduh": "ramai",
            "kontroversi": "perdebatan", "polemik": "perdebatan",
            "somasi": "teguran", "gugatan": "tuntutan",
           
            "reshuffle": "perombakan", "rotasi": "pergantian",
            "moratorium": "penghentian", "embargo": "larangan"
        }
        auto_synonyms.update(dataset_patterns)
        abbreviation_patterns = {
            "yg": "yang", "dg": "dengan", "krn": "karena", "utk": "untuk",
            "tdk": "tidak", "hrs": "harus", "sdh": "sudah", "blm": "belum",
            "dr": "dari", "ke": "ke", "pd": "pada", "ttg": "tentang",
            "spy": "supaya", "krg": "kurang", "lbh": "lebih"
        }
        auto_synonyms.update(abbreviation_patterns)
        return auto_synonyms
    except Exception as e:
        print(f"Error generating auto synonyms: {e}")
        return build_enhanced_synonym_dict_base()

def is_suspicious_change(input_text, matched_text):
    input_processed_for_diff = normalize_text(input_text)
    matched_processed_for_diff = normalize_text(matched_text)
    input_words = input_processed_for_diff.split()
    matched_words = matched_processed_for_diff.split()
    matcher = SequenceMatcher(None, input_words, matched_words)
    ratio = matcher.ratio()
    unique_input_words = set(input_words)
    unique_matched_words = set(matched_words)
    added_words = unique_input_words - unique_matched_words
    removed_words = unique_matched_words - unique_input_words
    total_unique_words = len(unique_input_words.union(unique_matched_words))
    diff_unique_words_count = len(added_words) + len(removed_words)
    prop_diff = diff_unique_words_count / total_unique_words if total_unique_words > 0 else 0
    if ratio < 0.98 and prop_diff > 0.15:
        return True, prop_diff
    return False, prop_diff

def is_negation_or_contradiction(text1, text2):
    negation_words = ['tidak', 'bukan', 'tak', 'belum', 'tanpa']
    tokens1 = set(normalize_text(text1).split())
    tokens2 = set(normalize_text(text2).split())
    for neg in negation_words:
        if (neg in tokens1 and neg not in tokens2) or (neg not in tokens1 and neg in tokens2):
            return True
    return False