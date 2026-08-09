// Menyediakan objek global agar terbebas dari error CORS/MIME tipe browser
window.MPTasksVision = {
    FilesetResolver: {
        forVisionTasks: async function(basePath) {
            // Mengarahkan pencarian langsung ke folder static utama tempat file .js dan .wasm berada
            return {
                wasmLoaderPath: "/static/vision_wasm_internal.js",
                wasmBinaryPath: "/static/vision_wasm_internal.wasm"
            };
        }
    },
    HandLandmarker: {
        createFromOptions: async function(vision, options) {
            console.log("Mock HandLandmarker lokal berhasil diaktifkan.");
            return {
                detectForVideo: function(video, timestamp) {
                    return { landmarks: [] };
                }
            };
        }
    }
};