

#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <sys/sysinfo.h>
#include <algorithm>
#include <numeric>

static const float PI      = 3.141592653589793f;
static const float EPSILON = 1e-10f;

static int FFT_SIZE                   = 1024;
static int NUM_FFTS_PER_BATCH         = 1000;
static int BATCH_SIZE                 = FFT_SIZE * NUM_FFTS_PER_BATCH;
static const int NUM_BUFFERS         = 3;
static const int TPB                 = 256;

enum WindowType { HAMMING = 0, BLACKMAN_HARRIS = 1, FLAT_TOP = 2, HANN = 3 };
static const char *WIN_NAMES[] = { "Hamming", "Blackman-Harris", "Flat-Top", "Hann" };

struct PipelineConfig {
    WindowType window_type    = BLACKMAN_HARRIS;
    float      sample_rate    = 10e6f;
    int        fft_size       = 1024;
    float      band_low_hz    = 0.8e6f;
    float      band_high_hz   = 4.8e6f;
    int        cfar_guard     = 4;
    int        cfar_ref       = 16;
    float      cfar_threshold = 10.0f;
    float      sparse_thresh  = -30.0f;
};

struct TimingResult {
    float total_ms   = 0.f;
    float window_ms  = 0.f;
    float fft_ms     = 0.f;
    float filter_ms  = 0.f;
    float mag_ms     = 0.f;
    float cfar_ms    = 0.f;
    int   cfar_total = 0;
    int   sparse_kept = 0;
};

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"[CUDA ERROR] %s — %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); \
    exit(1); } } while(0)
#define FFT_CHECK(x)  do { cufftResult r=(x); if(r!=CUFFT_SUCCESS){ \
    fprintf(stderr,"[cuFFT ERROR] code=%d — %s:%d\n",r,__FILE__,__LINE__); \
    exit(1); } } while(0)

__device__ __forceinline__
float windowCoeff(int n, int M, WindowType wt) {
    float x = (float)n, m = (float)M;
    switch (wt) {
        case HAMMING:
            return 0.54f - 0.46f * cosf(2.f*PI*x/m);
        case BLACKMAN_HARRIS:
            return 0.35875f - 0.48829f * cosf(2.f*PI*x/m)
                            + 0.14128f * cosf(4.f*PI*x/m)
                            - 0.01168f * cosf(6.f*PI*x/m);
        case FLAT_TOP:
            return 1.0f    - 1.93f  * cosf(2.f*PI*x/m)
                           + 1.29f  * cosf(4.f*PI*x/m)
                           - 0.388f * cosf(6.f*PI*x/m)
                           + 0.028f * cosf(8.f*PI*x/m);
        case HANN: default:
            return 0.5f * (1.f - cosf(2.f*PI*x/m));
    }
}

__global__ void applyWindow(cufftComplex* __restrict__ sig,
                             int N, int fft_size, WindowType wt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float w = windowCoeff(idx % fft_size, fft_size - 1, wt);
    sig[idx].x *= w;
    sig[idx].y *= w;
}

__global__ void magnitudeDB(const cufftComplex* __restrict__ fft,
                             float* __restrict__ mag, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float r = fft[idx].x, i = fft[idx].y;
    mag[idx] = 10.f * log10f(r*r + i*i + EPSILON);
}

__global__ void bandpassFilter(cufftComplex* __restrict__ fft,
                                int N, int fft_size,
                                float f_low, float f_high, float fs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float freq = (float)(idx % fft_size) * fs / (float)fft_size;
    if (freq < f_low || freq > f_high) {
        fft[idx].x = 0.f;
        fft[idx].y = 0.f;
    }
}

__global__ void cfarDetect(const float* __restrict__ spec,
                            int*         __restrict__ detect,
                            int N, int fft_size,
                            int guard, int ref, float threshold_db) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int bin  = idx % fft_size;
    int base = (idx / fft_size) * fft_size;

    if (bin < ref + guard || bin >= fft_size - ref - guard) {
        detect[idx] = 0;
        return;
    }

    float noise = 0.f;

    for (int k = bin - guard - ref; k < bin - guard; ++k)
        noise += spec[base + k];

    for (int k = bin + guard + 1; k <= bin + guard + ref; ++k)
        noise += spec[base + k];

    float noise_avg = noise / (float)(2 * ref);
    detect[idx] = (spec[idx] > noise_avg + threshold_db) ? 1 : 0;
}

__global__ void sparseMask(const float* __restrict__ spec,
                            int*         __restrict__ mask,
                            int N, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    mask[idx] = (spec[idx] > threshold) ? 1 : 0;
}

void load_signal_config(const char* bin_path, PipelineConfig& cfg) {
    std::string path(bin_path);
    size_t dot = path.find_last_of(".");
    std::string config_path = (dot != std::string::npos ? path.substr(0, dot) : path) + "_config.json";

    std::ifstream f(config_path);
    if (!f.is_open()) return;
    std::string line;
    while (std::getline(f, line)) {
        size_t colon = line.find(":");
        if (colon != std::string::npos) {
            if (line.find("sample_rate") != std::string::npos)
                cfg.sample_rate = std::stof(line.substr(colon + 1));
            if (line.find("fft_size") != std::string::npos)
                cfg.fft_size = std::stoi(line.substr(colon + 1));
            if (line.find("band_low_hz") != std::string::npos)
                cfg.band_low_hz = std::stof(line.substr(colon + 1));
            if (line.find("band_high_hz") != std::string::npos)
                cfg.band_high_hz = std::stof(line.substr(colon + 1));
        }
    }
    printf("[CONFIG] JSON chargé : %.1f Hz, FFT_SIZE=%d\n", cfg.sample_rate, cfg.fft_size);
}

int main(int argc, char** argv) {
    printf("GPU Radio Signal Processing\n\n");

    const char* input_file = (argc > 1) ? argv[1] : "signal.bin";

    int nDev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDev));
    for (int i = 0; i < nDev; i++) {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, i);
        printf("[GPU %d] %-24s | Arch %d.%d | VRAM %zu MB\n",
               i, p.name, p.major, p.minor, p.totalGlobalMem >> 20);
    }

    PipelineConfig cfg;
    load_signal_config(input_file, cfg);
    if (argc > 2) cfg.sample_rate = std::stof(argv[2]);

    FFT_SIZE = cfg.fft_size;

    if ((FFT_SIZE & (FFT_SIZE - 1)) != 0) {
        printf("[WARN] FFT_SIZE=%d n'est pas une puissance de 2, performances réduites\n", FFT_SIZE);
    }

    const int TARGET_BYTES_PER_SLOT = 256 * 1024 * 1024;
    size_t bytes_per_fft = (size_t)FFT_SIZE * (sizeof(cufftComplex) + sizeof(float) * 2 + sizeof(int) * 2);
    NUM_FFTS_PER_BATCH = std::min(512, std::max(1, (int)(TARGET_BYTES_PER_SLOT / bytes_per_fft)));
    BATCH_SIZE = FFT_SIZE * NUM_FFTS_PER_BATCH;

    printf("\n[CONFIG] Fichier cible   : %s\n",          input_file);
    printf("[CONFIG] Fenêtre         : %s\n",          WIN_NAMES[cfg.window_type]);
    printf("[CONFIG] Sample Rate     : %.1f Hz\n",     cfg.sample_rate);
    printf("[CONFIG] FFT Size        : %d\n",          FFT_SIZE);
    printf("[CONFIG] FFTs/Batch      : %d (VRAM optimisé)\n", NUM_FFTS_PER_BATCH);

    if (cfg.sample_rate < 1e6f) {
        cfg.band_low_hz = 0;
        cfg.band_high_hz = cfg.sample_rate / 2.0f;
    }

    printf("[CONFIG] Passe-bande     : %.2f — %.2f Hz\n",
           cfg.band_low_hz, cfg.band_high_hz);
    printf("[CONFIG] CFAR guard/ref  : %d / %d, seuil %.1f dB\n",
           cfg.cfar_guard, cfg.cfar_ref, cfg.cfar_threshold);
    printf("[CONFIG] Sparse seuil   : %.1f dB\n\n", cfg.sparse_thresh);

    printf("--- Chargement %s (streaming) ---\n", input_file);
    std::ifstream fin(input_file, std::ios::binary | std::ios::ate);
    if (!fin.is_open()) { fprintf(stderr,"[ERROR] %s introuvable\n", input_file); return 1; }
    size_t file_bytes = (size_t)fin.tellg();
    fin.seekg(0);
    size_t num_samples = (file_bytes / sizeof(float)) / 2;
    printf("[INFO] %zu IQ samples | %.1f MB\n", num_samples, file_bytes/1048576.0);

    printf("\nMode Full Streaming activé\n");

    size_t full_batches = num_samples / (size_t)BATCH_SIZE;
    size_t rem_samples  = num_samples % (size_t)BATCH_SIZE;
    size_t total_batches = full_batches + (rem_samples > 0 ? 1 : 0);
    printf("[INFO] %zu batches (%zu×%d + %zu restants)\n\n",
           total_batches, full_batches, BATCH_SIZE, rem_samples);

    cufftComplex* d_in [NUM_BUFFERS] = {};
    cufftComplex* d_fft[NUM_BUFFERS] = {};
    float*        d_mag[NUM_BUFFERS] = {};
    int*          d_det[NUM_BUFFERS] = {};
    int*          d_spr[NUM_BUFFERS] = {};

    cufftComplex* h_sig_batch [NUM_BUFFERS] = {};
    float*        h_spec_batch[NUM_BUFFERS] = {};
    int*          h_det_batch [NUM_BUFFERS] = {};
    int*          h_spr_batch [NUM_BUFFERS] = {};

    for (int b = 0; b < NUM_BUFFERS; b++) {
        CUDA_CHECK(cudaMalloc(&d_in [b], (size_t)BATCH_SIZE * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_fft[b], (size_t)BATCH_SIZE * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_mag[b], (size_t)BATCH_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_det[b], (size_t)BATCH_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_spr[b], (size_t)BATCH_SIZE * sizeof(int)));

        CUDA_CHECK(cudaMallocHost(&h_sig_batch [b], (size_t)BATCH_SIZE * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMallocHost(&h_spec_batch[b], (size_t)BATCH_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_det_batch [b], (size_t)BATCH_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_spr_batch [b], (size_t)BATCH_SIZE * sizeof(int)));
    }

    std::ofstream out_spec("output_spectrogram.bin", std::ios::binary);
    std::ofstream out_det ("output_detections.bin",  std::ios::binary);
    std::ofstream out_spr ("output_sparse.bin",      std::ios::binary);
    std::vector<int> bin_hits(FFT_SIZE, 0);

    printf("Pipeline asynchrone H2D ∥ Compute ∥ D2H\n\n");
    cudaStream_t s_h2d, s_cmp, s_d2h;
    CUDA_CHECK(cudaStreamCreate(&s_h2d));
    CUDA_CHECK(cudaStreamCreate(&s_cmp));
    CUDA_CHECK(cudaStreamCreate(&s_d2h));

    cudaEvent_t ev_h2d[NUM_BUFFERS], ev_cmp[NUM_BUFFERS], ev_d2h[NUM_BUFFERS];
    for (int b = 0; b < NUM_BUFFERS; b++) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_h2d[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_cmp[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_d2h[b], cudaEventDisableTiming));

        CUDA_CHECK(cudaEventRecord(ev_d2h[b], s_d2h));
    }

    int n_arr[] = { FFT_SIZE };
    cufftHandle plan_full;
    FFT_CHECK(cufftPlanMany(&plan_full, 1, n_arr,
                             nullptr, 1, FFT_SIZE, nullptr, 1, FFT_SIZE,
                             CUFFT_C2C, NUM_FFTS_PER_BATCH));
    FFT_CHECK(cufftSetStream(plan_full, s_cmp));

    int partial_ffts = (rem_samples > 0) ? (rem_samples / FFT_SIZE) : 0;
    cufftHandle plan_partial = 0;
    if (partial_ffts > 0) {
        FFT_CHECK(cufftPlanMany(&plan_partial, 1, n_arr,
                                 nullptr, 1, FFT_SIZE, nullptr, 1, FFT_SIZE,
                                 CUFFT_C2C, partial_ffts));
        FFT_CHECK(cufftSetStream(plan_partial, s_cmp));
    }

    cudaEvent_t t0, t1;
    cudaEvent_t tw0,tw1,tf0,tf1,tbp0,tbp1,tm0,tm1,tc0,tc1;
    CUDA_CHECK(cudaEventCreate(&t0));  CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&tw0)); CUDA_CHECK(cudaEventCreate(&tw1));
    CUDA_CHECK(cudaEventCreate(&tf0)); CUDA_CHECK(cudaEventCreate(&tf1));
    CUDA_CHECK(cudaEventCreate(&tbp0));CUDA_CHECK(cudaEventCreate(&tbp1));
    CUDA_CHECK(cudaEventCreate(&tm0)); CUDA_CHECK(cudaEventCreate(&tm1));
    CUDA_CHECK(cudaEventCreate(&tc0)); CUDA_CHECK(cudaEventCreate(&tc1));

    printf("--- Pipeline en cours ---\n");
    bool timing_captured = false;
    CUDA_CHECK(cudaEventRecord(t0));

    for (int iter = 0; iter < (int)total_batches + 2; iter++) {
        int b_h2d = iter;
        int b_cmp = iter - 1;
        int b_d2h = iter - 2;

        if (b_h2d >= 0 && b_h2d < (int)total_batches) {
            int buf    = b_h2d % NUM_BUFFERS;
            size_t cur = (b_h2d < (int)full_batches) ? (size_t)BATCH_SIZE : rem_samples;

            if (b_h2d >= NUM_BUFFERS)
                CUDA_CHECK(cudaStreamWaitEvent(s_h2d, ev_d2h[buf], 0));

            fin.read(reinterpret_cast<char*>(h_sig_batch[buf]), cur * sizeof(cufftComplex));

            CUDA_CHECK(cudaMemcpyAsync(d_in[buf], h_sig_batch[buf],
                cur * sizeof(cufftComplex), cudaMemcpyHostToDevice, s_h2d));
            CUDA_CHECK(cudaEventRecord(ev_h2d[buf], s_h2d));
        }

        if (b_cmp >= 0 && b_cmp < (int)total_batches) {
            int buf        = b_cmp % NUM_BUFFERS;
            int n_ffts     = (b_cmp < (int)full_batches) ? NUM_FFTS_PER_BATCH : (int)(rem_samples / FFT_SIZE);
            size_t padded  = (size_t)n_ffts * FFT_SIZE;
            int blocks     = (int)((padded + TPB - 1) / TPB);

            CUDA_CHECK(cudaStreamWaitEvent(s_cmp, ev_h2d[buf], 0));

            if (!timing_captured) CUDA_CHECK(cudaEventRecord(tw0, s_cmp));
            applyWindow<<<blocks, TPB, 0, s_cmp>>>(
                d_in[buf], padded, FFT_SIZE, cfg.window_type);
            if (!timing_captured) CUDA_CHECK(cudaEventRecord(tw1, s_cmp));

            if (!timing_captured) CUDA_CHECK(cudaEventRecord(tf0, s_cmp));
            cufftHandle plan = (b_cmp < (int)full_batches || plan_partial == 0)
                               ? plan_full : plan_partial;
            FFT_CHECK(cufftExecC2C(plan, d_in[buf], d_fft[buf], CUFFT_FORWARD));
            if (!timing_captured) CUDA_CHECK(cudaEventRecord(tf1, s_cmp));

            if (!timing_captured) CUDA_CHECK(cudaEventRecord(tbp0, s_cmp));
            bandpassFilter<<<blocks, TPB, 0, s_cmp>>>(
                d_fft[buf], padded, FFT_SIZE,
                cfg.band_low_hz, cfg.band_high_hz, cfg.sample_rate);
            if (!timing_captured) CUDA_CHECK(cudaEventRecord(tbp1, s_cmp));

            if (!timing_captured) CUDA_CHECK(cudaEventRecord(tm0, s_cmp));
            magnitudeDB<<<blocks, TPB, 0, s_cmp>>>(d_fft[buf], d_mag[buf], padded);
            if (!timing_captured) CUDA_CHECK(cudaEventRecord(tm1, s_cmp));

            if (!timing_captured) CUDA_CHECK(cudaEventRecord(tc0, s_cmp));
            cfarDetect<<<blocks, TPB, 0, s_cmp>>>(
                d_mag[buf], d_det[buf], padded, FFT_SIZE,
                cfg.cfar_guard, cfg.cfar_ref, cfg.cfar_threshold);
            if (!timing_captured) CUDA_CHECK(cudaEventRecord(tc1, s_cmp));

            sparseMask<<<blocks, TPB, 0, s_cmp>>>(
                d_mag[buf], d_spr[buf], padded, cfg.sparse_thresh);

            CUDA_CHECK(cudaEventRecord(ev_cmp[buf], s_cmp));
            timing_captured = true;
        }

        if (b_d2h >= 0 && b_d2h < (int)total_batches) {
            int buf       = b_d2h % NUM_BUFFERS;
            size_t offset = (size_t)b_d2h * BATCH_SIZE;
            size_t cur    = (b_d2h < (int)full_batches) ? (size_t)BATCH_SIZE : rem_samples;

            CUDA_CHECK(cudaStreamWaitEvent(s_d2h, ev_cmp[buf], 0));

            CUDA_CHECK(cudaMemcpyAsync(h_spec_batch[buf], d_mag[buf],
                cur * sizeof(float), cudaMemcpyDeviceToHost, s_d2h));
            CUDA_CHECK(cudaMemcpyAsync(h_det_batch[buf], d_det[buf],
                cur * sizeof(int), cudaMemcpyDeviceToHost, s_d2h));
            CUDA_CHECK(cudaMemcpyAsync(h_spr_batch[buf], d_spr[buf],
                cur * sizeof(int), cudaMemcpyDeviceToHost, s_d2h));

            CUDA_CHECK(cudaEventRecord(ev_d2h[buf], s_d2h));

            CUDA_CHECK(cudaStreamSynchronize(s_d2h));

            out_spec.write(reinterpret_cast<const char*>(h_spec_batch[buf]), cur * sizeof(float));
            out_det.write (reinterpret_cast<const char*>(h_det_batch[buf]),  cur * sizeof(int));

            for (size_t i = 0; i < cur; i++) {
                if (h_spr_batch[buf][i]) {
                    long long global_idx = (long long)(offset + i);
                    out_spr.write(reinterpret_cast<const char*>(&global_idx), sizeof(long long));
                    out_spr.write(reinterpret_cast<const char*>(&h_spec_batch[buf][i]), sizeof(float));
                }
                if (h_det_batch[buf][i]) {
                    bin_hits[(offset + i) % (size_t)FFT_SIZE]++;
                }
            }

            printf("[BATCH %3zu/%3zu] Streaming OK | Offset: %10zu | Samples: %8zu\n",
                   (size_t)b_d2h+1, total_batches, offset, cur);
            fflush(stdout);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    TimingResult tr;
    CUDA_CHECK(cudaEventElapsedTime(&tr.total_ms,  t0,   t1));
    CUDA_CHECK(cudaEventElapsedTime(&tr.window_ms, tw0,  tw1));
    CUDA_CHECK(cudaEventElapsedTime(&tr.fft_ms,    tf0,  tf1));
    CUDA_CHECK(cudaEventElapsedTime(&tr.filter_ms, tbp0, tbp1));
    CUDA_CHECK(cudaEventElapsedTime(&tr.mag_ms,    tm0,  tm1));
    CUDA_CHECK(cudaEventElapsedTime(&tr.cfar_ms,   tc0,  tc1));

    size_t num_ffts_total = std::max((size_t)1, num_samples / (size_t)FFT_SIZE);
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                   Détections CA-CFAR                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    float freq_res = cfg.sample_rate / FFT_SIZE;
    int det_count  = 0;
    for (int b = 0; b < FFT_SIZE; b++) {
        float pct = 100.f * bin_hits[b] / (float)num_ffts_total;
        if (pct > 30.f) {
            float freq_mhz = b * freq_res / 1e6f;
            printf("  [DÉTECTION] Bin %4d | %5.3f MHz | présent %.0f%% FFTs\n",
                   b, freq_mhz, pct);
            det_count++;
            tr.cfar_total++;
        }
    }
    if (det_count == 0) printf("  (Aucune détection persistante)\n");

    float throughput = (float)num_samples / tr.total_ms / 1000.f;
    printf("Analyse de Performance GPU\n");
    printf("Pipeline Total       : %7.2f ms\n", tr.total_ms);
    printf("Débit global         : %7.1f MSamples/sec\n", throughput);
    printf("Fenêtrage %-14s : %6.3f ms\n", WIN_NAMES[cfg.window_type], tr.window_ms);
    printf("cuFFT C2C Batch        : %6.3f ms\n", tr.fft_ms);
    printf("Filtre Passe-Bande     : %6.3f ms\n", tr.filter_ms);
    printf("Magnitude (dB)         : %6.3f ms\n", tr.mag_ms);
    printf("Détection CA-CFAR      : %6.3f ms\n", tr.cfar_ms);
    printf("\n");

    out_spec.close();
    out_det.close();
    out_spr.close();
    printf("[OK] Fichiers binaires sauvegardés incrémentalement.\n");

    cufftDestroy(plan_full);
    if (plan_partial) cufftDestroy(plan_partial);
    cudaStreamDestroy(s_h2d); cudaStreamDestroy(s_cmp); cudaStreamDestroy(s_d2h);
    for (int b = 0; b < NUM_BUFFERS; b++) {
        cudaFree(d_in[b]); cudaFree(d_fft[b]); cudaFree(d_mag[b]);
        cudaFree(d_det[b]); cudaFree(d_spr[b]);
        cudaFreeHost(h_sig_batch[b]);
        cudaFreeHost(h_spec_batch[b]);
        cudaFreeHost(h_det_batch[b]);
        cudaFreeHost(h_spr_batch[b]);
        cudaEventDestroy(ev_h2d[b]);
        cudaEventDestroy(ev_cmp[b]);
        cudaEventDestroy(ev_d2h[b]);
    }
    fin.close();
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaEventDestroy(tw0); cudaEventDestroy(tw1);
    cudaEventDestroy(tf0); cudaEventDestroy(tf1);
    cudaEventDestroy(tbp0); cudaEventDestroy(tbp1);
    cudaEventDestroy(tm0); cudaEventDestroy(tm1);
    cudaEventDestroy(tc0); cudaEventDestroy(tc1);

    printf("\n[SUCCESS] Pipeline GPU terminé avec succès !\n");
    return 0;
}
