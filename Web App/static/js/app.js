// --- 1. MEMASTIKAN BROWSER MEMBACA OBJECT DARI MEDIAPIPE_VISION.JS LOKAL ---
const MP = window.MPTasksVision || {};
const FilesetResolver = MP.FilesetResolver;
const HandLandmarker = MP.HandLandmarker;

const video = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const statusBadge = document.getElementById('status');

let listSukuKataAktif = []; 
let koordinatFrameBuffer = []; // Menampung sekuens frame koordinat untuk Model CRNN
const MAKS_FRAME_SEKUENS = 30; // Jumlah frame sekuens yang dibutuhkan Model 1 CRNN (Misal: 30 frame)

let handLandmarker = undefined;
let runningMode = "VIDEO";

// 2. Inisialisasi Hand Landmarker Menggunakan Jalur Folder Statis Lokal
async function initializeHandLandmarker() {
    try {
        if (!FilesetResolver || !HandLandmarker) {
            throw new Error("Library MediaPipe lokal belum siap di memori browser.");
        }

        const vision = await FilesetResolver.forVisionTasks("/");
        
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "/static/hand_landmarker.task", 
                delegate: "CPU"
            },
            runningMode: runningMode,
            numHands: 1
        });
        
        statusBadge.innerText = "Sistem SIBI Siap! Menyalakan Kamera...";
        statusBadge.style.backgroundColor = "#2ecc71"; 
        
        startCamera();
    } catch (error) {
        console.warn("Memicu bypass darurat kamera akibat kendala engine:", error);
        statusBadge.innerText = "Mode Bypass Active (Model Off)";
        statusBadge.style.backgroundColor = "#f39c12";
        handLandmarker = { detectForVideo: () => ({ landmarks: [] }) };
        startCamera();
    }
}

// 3. Akses Webcam Laptop (Auto-Play)
function startCamera() {
    const constraints = { 
        video: { width: 640, height: 480, facingMode: "user" } 
    };

    navigator.mediaDevices.getUserMedia(constraints)
    .then((stream) => {
        video.srcObject = stream;
        video.play().then(() => {
            console.log("Webcam aktif.");
            predictWebcam(); 
        });
    })
    .catch((err) => {
        console.error("Gagal membuka kamera: ", err);
        statusBadge.innerText = "Error: Kamera diblokir";
        statusBadge.style.backgroundColor = "#e74c3c";
    });
}

// 4. Loop Deteksi Koordinat Nyata & Pengumpulan Sekuens Frame
let lastVideoTime = -1;
async function predictWebcam() {
    if (video.videoWidth > 0) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }
    
    let startTimeMs = Date.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        
        if (handLandmarker && typeof handLandmarker.detectForVideo === 'function') {
            const results = handLandmarker.detectForVideo(video, startTimeMs);
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            
            if (results && results.landmarks && results.landmarks.length > 0) {
                const landmarks = results.landmarks[0];
                drawHandSkeleton(landmarks);
                
                // --- EKSTRAKSI TITIK KOORDINAT (X, Y, Z) MENJADI FLAT ARRAY ---
                // 21 titik landmarks dikali (x, y, z) menghasilkan total 63 fitur numerik per frame.
                let frameFeatures = [];
                landmarks.forEach(lm => {
                    frameFeatures.push(lm.x, lm.y, lm.z);
                });
                
                // Masukkan data fitur frame saat ini ke dalam antrean sekuens buffer
                koordinatFrameBuffer.push(frameFeatures);
                
                // Jika kumpulan frame sudah terkumpul sesuai panjang sekuens Model CRNN (30 frame)
                if (koordinatFrameBuffer.length === MAKS_FRAME_SEKUENS) {
                    kirimKeModelCRNNGerakan(koordinatFrameBuffer);
                    koordinatFrameBuffer = []; // Kosongkan kembali buffer untuk menangkap gerakan kata selanjutnya
                }
            } else {
                // Jika tangan keluar dari jangkauan kamera, kurangi tumpukan frame perlahan agar tidak macet
                if (koordinatFrameBuffer.length > 0) {
                    koordinatFrameBuffer.shift();
                }
            }
        }
    }
    window.requestAnimationFrame(predictWebcam);
}

// 5. Menggambar Struktur Tulang Tangan di Web Canvas
function drawHandSkeleton(landmarks) {
    canvasCtx.fillStyle = "#2ecc71";
    canvasCtx.strokeStyle = "#ffffff";
    canvasCtx.lineWidth = 3;
    for (const landmark of landmarks) {
        const x = landmark.x * canvasElement.width;
        const y = landmark.y * canvasElement.height;
        canvasCtx.beginPath();
        canvasCtx.arc(x, y, 5, 0, 2 * Math.PI);
        canvasCtx.fill();
    }
}

// 6. [KIRIM DATA ASLI 1] Kirim Sekuens Koordinat Tangan ke Model CRNN di Flask
function kirimKeModelCRNNGerakan(sekuensKoordinat) {
    fetch('/predict_gesture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: sekuensKoordinat }) // TYPO FIX: variabel pengiriman sudah disamakan
    })
    .then(response => response.json())
    .then(data => {
        // Abaikan jika model memprediksi karakter kosong "_" atau tidak mengenali gerakan
        if (data.status === 'success' && data.suku_kata_terdeteksi && data.suku_kata_terdeteksi !== "_") {
            
            // Masukkan hasil prediksi suku kata asli ke dalam list suku kata aktif di layar
            listSukuKataAktif.push(data.suku_kata_terdeteksi);
            
            // Batasi panjang suku kata yang ditampung (maksimal 5 suku kata sesuai batas padding maxlen RNN)
            if (listSukuKataAktif.length > 5) {
                listSukuKataAktif.shift(); 
            }
            
            document.getElementById('list-suku-kata').innerText = JSON.stringify(listSukuKataAktif);
            
            // Panggil Model 2 untuk menerjemahkan rangkaian suku kata aktif tersebut menjadi kata dasar utuh
            kirimKeModelRNNTeks(listSukuKataAktif);
        }
    })
    .catch(error => console.error("Gagal memanggil Model CRNN Gerakan di Flask:", error));
}

// 7. [KIRIM DATA ASLI 2] Kirim List Suku Kata Terkumpul ke Model RNN Teks di Flask
function kirimKeModelRNNTeks(arraySukuKata) {
    fetch('/predict_text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ suku_kata: arraySukuKata })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            document.getElementById('kata-hasil').innerText = data.kata_hasil_terjemahan;
        }
    })
    .catch(error => console.error("Gagal memanggil Model RNN Teks di Flask:", error));
}

// Jalankan Inisialisasi Utama saat halaman dimuat
initializeHandLandmarker();