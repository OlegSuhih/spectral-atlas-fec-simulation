# Photonic Spectral Atlas: Hamming(7,4) FEC Simulation

This project implements a numerical simulation of a **Spectral Atlas** photonic processor pipeline. It evaluates the integration of **InAs/InGaAs Quantum Dot (QD)** hardware characteristics with a **Hamming(7,4) Soft-Decision Forward Error Correction (FEC)** scheme.

## Key Research Components

### 1. Photonic Hardware Modeling
* Simulation of gain and noise profiles for **InAs/InGaAs Quantum Dots**.
* Analysis of optical signal degradation within the photonic atlas architecture.

### 2. Error Correction Pipeline
* Implementation of **Hamming(7,4)** algorithm.
* Focus on **Soft-Decision decoding**, which utilizes reliability information to improve Bit Error Rate (BER) compared to traditional hard-decision methods.

### 3. Scalability & Architecture
The simulation is designed with an expandable logic to support:
* **Multi-sector space mapping** (Sectors containing live/dead planets).
* **Granular resource accounting** using unique ID structures.

## Usage

Ensure you have Python installed with the following dependencies:
* `numpy`
* `matplotlib`

Run the simulation:
```bash
python spectral_atlas_hamming_fec.py
```

## Results & Analysis
The simulation provides a comparative analysis of data integrity before and after the FEC pipeline, demonstrating the efficiency of the spectral atlas approach in high-speed optical computing environments.
