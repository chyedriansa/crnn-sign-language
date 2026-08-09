from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# --- KONFIGURASI PATH FOLDER ---
BASE_TRAIN_DIR = os.path.join('..', 'hasil_pelatihan')
MODEL_TEXT_PATH = os.path.join(BASE_TRAIN_DIR, 'model_rnn_suku_kata_terbaik.h5')
MODEL_GESTURE_PATH = os.path.join(BASE_TRAIN_DIR, 'model_crnn_sibi_terbaik.h5')
VOCAB_PATH = os.path.join(BASE_TRAIN_DIR, 'syllable_to_idx.json')
CSV_PATH = os.path.join(BASE_TRAIN_DIR, 'data_kata_dasar_rnn_syllable_training.csv')

# --- 1. MEMUAT KAMUS INPUT SUKU KATA (SUKU KATA -> INDEX ANGKA) ---
print("Memuat kamus suku kata (syllable_to_idx.json)...")
with open(VOCAB_PATH, 'r') as f:
    syllable_to_idx = json.load(f)

# Membalik kamus untuk pemetaan output Model CRNN (Index Angka -> String Suku Kata)
idx_to_syllable = {int(v): k for k, v in syllable_to_idx.items()}

# --- 2. EKSTRAKSI OTOMATIS KAMUS OUTPUT KATA DASAR (INDEX ANGKA -> STR KATA DASAR) ---
print("Mengekstrak daftar kata dasar dari file CSV...")
df_csv = pd.read_csv(CSV_PATH)
kolom_target = 'target_word'
df_csv['kata_dasar_clean'] = df_csv[kolom_target].astype(str).str.strip().str.lower()
unique_words = sorted(df_csv['kata_dasar_clean'].unique())
idx_to_word = {idx: word for idx, word in enumerate(unique_words)}
print(f"--- [SUKSES] Berhasil memetakan {len(idx_to_word)} target kata dasar SIBI! ---")

# --- 3. MEMUAT KEDUA MODEL KECERDASAN BUATAN (.H5) ---
print("Memuat Model 1: CRNN Gerakan Tangan SIBI...")
model_gesture = tf.keras.models.load_model(MODEL_GESTURE_PATH)

print("Memuat Model 2: RNN Penerjemah Suku Kata ke Teks...")
model_text = tf.keras.models.load_model(MODEL_TEXT_PATH)
print("--- [SUKSES] Kedua Model Berhasil Dimuat ke Flask! ---")


# --- ROUTE KHUSUS UNTUK FILE MODEL .TASK (MENGATASI BLOCK CORS BROWSER) ---
@app.route('/static/hand_landmarker.task')
def serve_hand_landmarker():
    return send_from_directory(os.path.join(app.root_path, 'static'), 
                               'hand_landmarker.task', 
                               mimetype='application/octet-stream')

# --- ROUTE 1: HALAMAN UTAMA WEB ---
@app.route('/')
def index():
    return render_template('index.html')


# --- ROUTE 2: API ENDPOINT UNTUK MODEL 1 (PREDIKSI GERAKAN TANGAN MENJADI SUKU KATA) ---
@app.route('/predict_gesture', methods=['POST'])
def predict_gesture():
    try:
        data = request.get_json()
        landmarks_sequence = data.get('sequence', [])
        
        if not landmarks_sequence:
            return jsonify({'status': 'error', 'message': 'Sequence koordinat kosong'}), 400
            
        # Mengubah data list dari JavaScript menjadi numpy array sesuai dimensi input shape Model CRNN
        # Dimensinya: (batch_size=1, timesteps=30, features=63)
        input_array = np.array([landmarks_sequence], dtype=np.float32)
        
        # Eksekusi prediksi menggunakan model_crnn_sibi_terbaik.h5
        predictions = model_gesture.predict(input_array, verbose=0)
        predicted_idx = np.argmax(predictions, axis=-1)[0]
        
        # Ambil nama suku kata berdasarkan indeks kelas tertinggi hasil prediksi
        suku_kata_terpilih = idx_to_syllable.get(int(predicted_idx), "_")
        
        return jsonify({
            'status': 'success',
            'suku_kata_terdeteksi': suku_kata_terpilih
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# --- ROUTE 3: API ENDPOINT UNTUK MODEL 2 (PREDIKSI KUMPULAN SUKU KATA MENJADI KATA UTUH) ---
@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        data = request.get_json()
        input_suku_kata = data.get('suku_kata', [])
        
        if not input_suku_kata:
            return jsonify({'status': 'error', 'message': 'Input suku kata kosong'}), 400
            
        encoded_sequence = [syllable_to_idx.get(sk.lower(), 0) for sk in input_suku_kata]
        padded_sequence = pad_sequences([encoded_sequence], maxlen=5, padding='post')
        
        predictions = model_text.predict(padded_sequence, verbose=0)
        predicted_idx = np.argmax(predictions, axis=-1)[0]
        kata_terjemahan = idx_to_word.get(predicted_idx, "Kata Tidak Dikenali")
        
        return jsonify({
            'status': 'success',
            'input_diterima': input_suku_kata,
            'kata_hasil_terjemahan': kata_terjemahan
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)