# Spectral Atlas

Simulation of a deep photonic computing pipeline based on InAs/InGaAs quantum dots with spectral encoding and Hamming(7,4) soft-decision forward error correction.

## Energy Model

- Base energy per node: **84 aJ**
- Energy per node grows with depth: `84 × 1.025^n` aJ (n = node index)
- Total energy = sum across all nodes
- Example (24-node hardcore test): **Total energy ≈ 2706 aJ** → **Average ≈ 113 aJ per node**

## Key Capabilities Demonstrated

- Reliable propagation over **20–24 nodes** even when SNR drops below 1 dB (and sometimes below 0 dB)
- Hamming(7,4) soft-decision FEC with maximum-likelihood decoding
- Resilience to AWGN + occasional burst noise (15% probability, 3.5× amplitude spike)
- Nonlinear SNR degradation along the pipeline

## Files

- `spectral_atlas.py` — main simulation with several test scenarios
- `spectral_atlas_hardcore_test.py` — extreme 30-node stress test with burst noise

## How to Run

```bash
pip install numpy matplotlib
python spectral_atlas.py
python spectral_atlas_hardcore_test.py
```

## Purpose
This repository shows that soft-decision FEC can significantly extend the usable depth of a noisy photonic pipeline while keeping energy consumption in the low attojoule range per node.
The code is kept simple and fully reproducible for easy verification and further development.

## License

MIT License — free for academic and commercial use with attribution.
