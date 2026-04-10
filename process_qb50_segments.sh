#!/bin/bash
set -e

WAV_FILE="qb50-436500kHz-2017-05-29-182529.wav"
SEGMENT_DURATION=300
NUM_SEGMENTS=5 

echo "=========================================="
echo "Traitement QB50 par segments ($SEGMENT_DURATION s)"
echo "=========================================="

for SEG in $(seq 1 $NUM_SEGMENTS); do
    START=$(( ($SEG - 1) * $SEGMENT_DURATION ))
    DURATION=$SEGMENT_DURATION
    
   
    if [ $SEG -eq $NUM_SEGMENTS ]; then
        DURATION=$(( 1388 - $START ))
    fi
    
    echo ""
    echo "[SEGMENT $SEG/$NUM_SEGMENTS] $START → $(($START + $DURATION))s"
    
   
    python3 convert_satnogs.py "$WAV_FILE" signal_seg${SEG}.bin \
        --start $START --duration $DURATION
    
   
    cp signal_seg${SEG}_config.json signal_config.json
    
   
    ./build/radio_fft signal_seg${SEG}.bin
    
   
    python3 plot_spectrogram.py
    
   
    mv captures/spectrogram.png captures/spectrogram_seg${SEG}.png
    mv captures/spectrogram_doppler.png captures/spectrogram_doppler_seg${SEG}.png
    mv captures/spectrum_avg.png captures/spectrum_avg_seg${SEG}.png
    
   
    rm -f signal_seg${SEG}.bin signal_seg${SEG}_config.json
    
    echo "[OK] Segment $SEG terminé"
done

echo ""
echo "=========================================="
echo "Tous les segments traités !"
echo "=========================================="
echo ""
echo "Résultats générés:"
ls -lh spectrogram_seg*.png spectrum_avg_seg*.png 2>/dev/null | wc -l
echo "fichiers PNG"
