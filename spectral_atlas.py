import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# ==================== SPECTRAL PROFILES ====================

def spectral_profile(frequencies: np.ndarray, center_freq: float, width: float = 0.3, intensity: float = 1.0) -> np.ndarray:
    """Gaussian spectral profile of a photon"""
    return intensity * np.exp(-((frequencies - center_freq) ** 2) / (2 * width ** 2))


def create_spectral_atlas(num_symbols: int = 16, freq_start: float = 1.0, freq_end: float = 8.0, 
                          freq_points: int = 128, width: float = 0.25) -> Tuple[Dict[int, np.ndarray], np.ndarray, list]:
    """Creates spectral atlas with unique profiles for each symbol"""
    frequencies = np.linspace(freq_start, freq_end, freq_points)
    atlas = {}
    center_freqs = []
    
    for i in range(num_symbols):
        center = freq_start + (i / (num_symbols - 1)) * (freq_end - freq_start)
        center_freqs.append(center)
        atlas[i] = spectral_profile(frequencies, center, width=width)
    
    return atlas, frequencies, center_freqs


def bits_to_symbol(bits: np.ndarray) -> int:
    """Converts 4 bits to symbol index (0-15)"""
    return int(''.join(map(str, bits.astype(int))), 2)


def encode_to_spectrum(bits: np.ndarray, atlas: Dict[int, np.ndarray]) -> np.ndarray:
    """Encodes 4 bits into spectral profile"""
    return atlas[bits_to_symbol(bits)].copy()


# ==================== PHOTONIC CHANNEL ====================

def spectral_interference(spectrum1: np.ndarray, spectrum2: np.ndarray, 
                          freq_distance: float, coherence: float = 0.8) -> np.ndarray:
    """Coherent spectral interference of two photons"""
    strength = coherence * np.exp(-freq_distance ** 2 / 0.5)
    phase = np.exp(1j * np.random.uniform(0, 2 * np.pi))
    return spectrum1 + strength * np.abs(phase) * spectrum2


def apply_channel_effects(spectrum: np.ndarray, snr_db: float, 
                          dispersion: float = 0.03, absorption: float = 0.02) -> np.ndarray:
    """Applies absorption, dispersion, and AWGN noise"""
    # Absorption
    absorbed = spectrum * np.exp(-absorption)
    
    # Dispersion (Gaussian smearing)
    if dispersion > 0:
        kernel_size = max(3, int(dispersion * len(spectrum)))
        kernel = np.exp(-((np.arange(kernel_size) - kernel_size//2) ** 2) / (2 * (dispersion * 10) ** 2))
        kernel = kernel / kernel.sum()
        dispersed = np.convolve(absorbed, kernel, mode='same')
    else:
        dispersed = absorbed
    
    # AWGN noise
    sigma = 10 ** (-snr_db / 20)
    noise = np.random.normal(0, sigma * np.max(dispersed), len(dispersed))
    
    result = dispersed + noise
    return np.maximum(result, 0)  # intensity cannot be negative


# ==================== DECODING ====================

def decode_spectral_mle(received_spectrum: np.ndarray, atlas: Dict[int, np.ndarray]) -> Tuple[np.ndarray, float]:
    """Maximum Likelihood decoding via correlation with atlas"""
    best_idx = -1
    best_corr = -np.inf
    
    for idx, profile in atlas.items():
        corr = np.sum(received_spectrum * profile)
        corr /= (np.linalg.norm(received_spectrum) * np.linalg.norm(profile) + 1e-9)
        if corr > best_corr:
            best_corr = corr
            best_idx = idx
    
    bits = np.array([int(b) for b in format(best_idx, '04b')])
    return bits, best_corr


# ==================== SIMULATION ====================

def run_spectral_atlas_simulation(input_bits: Tuple[int, int, int, int] = (1, 0, 1, 0),
                                   initial_snr: float = 20.0,
                                   nodes: int = 25,
                                   burst_probability: float = 0.15,
                                   enable_interference: bool = True,
                                   enable_dispersion: bool = True,
                                   verbose: bool = True) -> dict:
    """Main simulation of Spectral Atlas photonic system"""
    
    atlas, frequencies, center_freqs = create_spectral_atlas()
    orig_bits = np.array(input_bits)
    original_spectrum = encode_to_spectrum(orig_bits, atlas)
    
    current_snr = initial_snr
    correct = 0
    energies, correlations, snr_history, errors = [], [], [], []
    
    if verbose:
        print("\n" + "="*90)
        print(f" SPECTRAL ATLAS | Start SNR: {initial_snr} dB | Target: {orig_bits}")
        print("="*90)
        print(f"{'Node':<6} | {'SNR (dB)':<10} | {'Corr':<8} | {'Status'}")
        print("-"*70)
    
    for node in range(nodes):
        # SNR degradation
        drop = 0.7 + (node * 0.06) ** 1.8
        current_snr -= drop
        snr_history.append(current_snr)
        
        received = original_spectrum.copy()
        
        # Channel effects
        disp = 0.03 if enable_dispersion else 0
        received = apply_channel_effects(received, current_snr, disp)
        
        # Interference
        if enable_interference and np.random.rand() < burst_probability:
            neighbor = np.random.randint(0, len(atlas))
            dist = abs(center_freqs[neighbor] - center_freqs[bits_to_symbol(orig_bits)])
            received = spectral_interference(received, atlas[neighbor], dist)
        
        # Decode
        decoded, corr = decode_spectral_mle(received, atlas)
        success = np.array_equal(decoded, orig_bits)
        
        # Energy
        energy = int(50 * (1.02 ** node) * (1 + max(0, (20 - current_snr) / 40)))
        energies.append(energy)
        correlations.append(corr)
        
        if success:
            correct += 1
        else:
            errors.append(node)
        
        if verbose:
            status = "✓" if success else "✗"
            print(f"{node:<6} | {current_snr:<10.2f} | {corr:<8.4f} | {status}")
        
        if current_snr < -10:
            if verbose:
                print(f"\n⚠️ Signal lost at SNR={current_snr:.2f} dB")
            break
    
    total = node + 1
    if verbose:
        print("-"*70)
        print(f"\n✅ Nodes passed: {correct}/{total} ({correct/total*100:.1f}%)")
        print(f"⚡ Total energy: {sum(energies)} aJ")
        print(f"📈 Avg correlation: {np.mean(correlations):.4f}")
        print("="*90 + "\n")
    
    return {
        'success_rate': correct / total,
        'correct_count': correct,
        'total_nodes': total,
        'total_energy': sum(energies),
        'correlations': correlations,
        'snr_history': snr_history,
        'errors_by_node': errors
    }


# ==================== VISUALIZATION ====================

def visualize_spectral_atlas(atlas: Dict[int, np.ndarray], frequencies: np.ndarray, 
                             save_path: Optional[str] = None):
    """Plot the spectral atlas"""
    plt.figure(figsize=(12, 6))
    for idx, profile in atlas.items():
        plt.plot(frequencies, profile, alpha=0.6, lw=1, label=f'S{idx}')
    plt.xlabel('Frequency')
    plt.ylabel('Intensity')
    plt.title('Spectral Atlas: 16 Photonic Signatures')
    plt.legend(ncol=4, fontsize=8)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_simulation_results(results: dict):
    """Plot simulation results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # SNR degradation
    axes[0, 0].plot(results['snr_history'], 'b-', lw=2)
    axes[0, 0].axhline(0, color='r', ls='--', alpha=0.5, label='Noise floor')
    axes[0, 0].set_xlabel('Node')
    axes[0, 0].set_ylabel('SNR (dB)')
    axes[0, 0].set_title('Signal Degradation')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Correlation
    axes[0, 1].plot(results['correlations'], 'g-', lw=2)
    axes[0, 1].axhline(0.8, color='orange', ls='--', alpha=0.5, label='Threshold')
    axes[0, 1].set_xlabel('Node')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_title('Decoding Confidence')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Errors
    ax3 = axes[1, 0]
    err = np.zeros(results['total_nodes'])
    for n in results['errors_by_node']:
        err[n] = 1
    ax3.bar(range(results['total_nodes']), err, color='red', alpha=0.7)
    ax3.set_xlabel('Node')
    ax3.set_ylabel('Error')
    ax3.set_title('Error Distribution')
    ax3.set_ylim(0, 1.2)
    ax3.grid(True, alpha=0.3)
    
    # Success rate
    ax4 = axes[1, 1]
    sr = results['success_rate']
    ax4.pie([sr, 1-sr], labels=[f'PASS ({sr*100:.1f}%)', f'FAIL'], 
            colors=['green', 'red'], autopct='%1.1f%%')
    ax4.set_title('Overall Performance')
    
    plt.tight_layout()
    plt.show()


# ==================== STRESS TEST ====================

def run_stress_test_scenarios():
    """Run multiple test scenarios"""
    print("\n🔥 SPECTRAL ATLAS STRESS TEST SUITE 🔥")
    
    scenarios = [
        ("BASELINE", 20, 25, False, False),
        ("INTERFERENCE", 20, 25, True, False),
        ("FULL PHYSICS", 20, 25, True, True),
        ("LOW SNR", 15, 20, True, True),
        ("EXTREME", 18, 35, True, True),
    ]
    
    results = []
    for name, snr, n, inter, disp in scenarios:
        print(f"\n📡 {name}: SNR={snr}dB, Nodes={n}, Interference={inter}, Dispersion={disp}")
        r = run_spectral_atlas_simulation(initial_snr=snr, nodes=n, 
                                          enable_interference=inter, enable_dispersion=disp, verbose=True)
        results.append((name, r['success_rate'], r['total_energy']))
    
    print("\n📊 SUMMARY")
    print(f"{'Scenario':<15} | {'Success':<10} | {'Energy (aJ)'}")
    print("-"*45)
    for name, sr, eng in results:
        print(f"{name:<15} | {sr*100:>5.1f}%{'':<4} | {eng}")
    
    return results


# ==================== MAIN ====================

if __name__ == "__main__":
    print("🔬 CREATING SPECTRAL ATLAS...")
    atlas, freqs, _ = create_spectral_atlas()
    visualize_spectral_atlas(atlas, freqs)
    
    print("\n🚀 RUNNING SIMULATION...")
    results = run_spectral_atlas_simulation(nodes=30, verbose=True)
    
    visualize_simulation_results(results)
    
    print("\n⚡ RUNNING STRESS TESTS...")
    run_stress_test_scenarios()
    
    print(f"\n✨ DONE | Performance: {results['success_rate']*100:.1f}% over {results['total_nodes']} nodes")
    print(f"   Total energy: {results['total_energy']} aJ | Avg correlation: {np.mean(results['correlations']):.3f}")
