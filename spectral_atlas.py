import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# ==================== СПЕКТРАЛЬНІ ПРОФІЛІ ====================

def spectral_profile(frequencies: np.ndarray, center_freq: float, width: float = 0.3, intensity: float = 1.0) -> np.ndarray:
    """
    Гауссів спектральний профіль фотона
    
    Args:
        frequencies: масив частот
        center_freq: центральна частота (несе інформацію)
        width: ширина спектра (чим менше, тим "чистіший" фотон)
        intensity: пікова інтенсивність
    """
    return intensity * np.exp(-((frequencies - center_freq) ** 2) / (2 * width ** 2))


def create_spectral_atlas(num_symbols: int = 16, freq_start: float = 1.0, freq_end: float = 8.0, 
                          freq_points: int = 128, width: float = 0.25) -> Tuple[Dict[int, np.ndarray], np.ndarray, list]:
    """
    Створює спектральний атлас — набір унікальних спектральних профілів
    
    Returns:
        atlas: словник {індекс символу: спектральний профіль}
        frequencies: масив частот
        center_freqs: список центральних частот для кожного символу
    """
    frequencies = np.linspace(freq_start, freq_end, freq_points)
    atlas = {}
    center_freqs = []
    
    for i in range(num_symbols):
        # Рівномірний розподіл центральних частот
        center = freq_start + (i / (num_symbols - 1)) * (freq_end - freq_start)
        center_freqs.append(center)
        atlas[i] = spectral_profile(frequencies, center, width=width)
    
    return atlas, frequencies, center_freqs


def bits_to_symbol(bits: np.ndarray) -> int:
    """Перетворює 4 біти в індекс символу (0-15)"""
    return int(''.join(map(str, bits.astype(int))), 2)


def encode_to_spectrum(bits: np.ndarray, atlas: Dict[int, np.ndarray]) -> np.ndarray:
    """Кодує 4 біти в спектральний профіль"""
    symbol_idx = bits_to_symbol(bits)
    return atlas[symbol_idx].copy()


# ==================== ФІЗИКА ФОТОННОГО КАНАЛУ ====================

def spectral_interference(spectrum1: np.ndarray, spectrum2: np.ndarray, 
                          freq_distance: float, coherence: float = 0.8) -> np.ndarray:
    """
    Когерентна спектральна інтерференція двох фотонів
    
    Args:
        spectrum1, spectrum2: спектральні профілі
        freq_distance: відстань між центральними частотами
        coherence: ступінь когерентності (0-1)
    """
    # Чим ближче спектри, тим сильніша інтерференція
    interference_strength = coherence * np.exp(-freq_distance ** 2 / 0.5)
    
    # Випадкова фаза (реалістична для фотонних систем)
    phase = np.exp(1j * np.random.uniform(0, 2 * np.pi))
    
    # Інтерферований спектр (інтенсивність)
    return spectrum1 + interference_strength * np.abs(phase) * spectrum2


def apply_channel_effects(spectrum: np.ndarray, snr_db: float, 
                          dispersion: float = 0.03, absorption: float = 0.02) -> np.ndarray:
    """
    Застосовує фізичні ефекти фотонного каналу
    
    Args:
        spectrum: вхідний спектральний профіль
        snr_db: відношення сигнал/шум в dB
        dispersion: коефіцієнт дисперсії (розмиття спектра)
        absorption: коефіцієнт поглинання
    
    Returns:
        спотворений спектральний профіль
    """
    # 1. Поглинання (експоненційне загасання)
    absorbed = spectrum * np.exp(-absorption)
    
    # 2. Дисперсія — розмиття спектра (згортка з гауссовим ядром)
    if dispersion > 0:
        kernel_size = max(3, int(dispersion * len(spectrum)))
        kernel = np.exp(-((np.arange(kernel_size) - kernel_size//2) ** 2) / (2 * (dispersion * 10) ** 2))
        kernel = kernel / kernel.sum()
        dispersed = np.convolve(absorbed, kernel, mode='same')
    else:
        dispersed = absorbed
    
    # 3. Спектральний шум (AWGN)
    sigma = 10 ** (-snr_db / 20)
    noise = np.random.normal(0, sigma * np.max(dispersed), len(dispersed))
    
    # 4. Нормалізація (збереження енергії)
    result = dispersed + noise
    result = np.maximum(result, 0)  # фізично: інтенсивність не може бути негативною
    
    return result


# ==================== ДЕКОДУВАННЯ ====================

def decode_spectral_mle(received_spectrum: np.ndarray, atlas: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Maximum Likelihood Estimation (MLE) декодування через кореляцію з атласом
    
    Returns:
        декодовані 4 біти
    """
    best_idx = -1
    best_correlation = -np.inf
    
    for idx, profile in atlas.items():
        # Нормалізована взаємна кореляція
        correlation = np.sum(received_spectrum * profile)
        correlation /= (np.linalg.norm(received_spectrum) * np.linalg.norm(profile) + 1e-9)
        
        if correlation > best_correlation:
            best_correlation = correlation
            best_idx = idx
    
    # Повертаємо біти
    bits = np.array([int(b) for b in format(best_idx, '04b')])
    return bits, best_correlation


# ==================== ОСНОВНА СИМУЛЯЦІЯ ====================

def run_spectral_atlas_simulation(input_bits: Tuple[int, int, int, int] = (1, 0, 1, 0),
                                   initial_snr: float = 20.0,
                                   nodes: int = 25,
                                   burst_probability: float = 0.15,
                                   enable_interference: bool = True,
                                   enable_dispersion: bool = True,
                                   verbose: bool = True) -> dict:
    """
    Повна симуляція фотонної системи Spectral Atlas
    
    Returns:
        словник з результатами симуляції
    """
    # Ініціалізація спектрального атласу
    atlas, frequencies, center_freqs = create_spectral_atlas()
    orig_bits = np.array(input_bits)
    original_spectrum = encode_to_spectrum(orig_bits, atlas)
    
    # Параметри симуляції
    current_snr = initial_snr
    correct_count = 0
    node_energies = []
    correlations = []
    snr_history = []
    errors_by_node = []
    
    if verbose:
        print("\n" + "="*90)
        print(f" SPECTRAL ATLAS SIMULATION | Start SNR: {initial_snr} dB | Target: {orig_bits}")
        print("="*90)
        print(f"{'Node':<6} | {'SNR (dB)':<10} | {'Correlation':<12} | {'Status':<8} | {'Interference'}")
        print("-"*90)
    
    for node in range(nodes):
        # 1. Деградація SNR (нелінійна)
        drop = 0.7 + (node * 0.06) ** 1.8
        current_snr -= drop
        snr_history.append(current_snr)
        
        # 2. Копія сигналу для цього вузла
        received_spectrum = original_spectrum.copy()
        
        # 3. Ефекти каналу
        dispersion = 0.03 if enable_dispersion else 0
        received_spectrum = apply_channel_effects(received_spectrum, current_snr, dispersion)
        
        # 4. Інтерференція від сусідніх каналів
        interference_msg = ""
        if enable_interference and np.random.rand() < burst_probability:
            # Випадковий сусідній канал
            neighbor_idx = np.random.randint(0, len(atlas))
            neighbor_spectrum = atlas[neighbor_idx]
            freq_distance = abs(center_freqs[neighbor_idx] - center_freqs[bits_to_symbol(orig_bits)])
            received_spectrum = spectral_interference(received_spectrum, neighbor_spectrum, freq_distance)
            interference_msg = f"from ch{neighbor_idx}"
        
        # 5. Декодування
        decoded_bits, correlation = decode_spectral_mle(received_spectrum, atlas)
        success = np.array_equal(decoded_bits, orig_bits)
        
        # 6. Енергоспоживання вузла (зростає з деградацією)
        node_energy = int(50 * (1.02 ** node) * (1 + max(0, (20 - current_snr) / 40)))
        node_energies.append(node_energy)
        correlations.append(correlation)
        
        # 7. Статистика
        if success:
            correct_count += 1
        else:
            errors_by_node.append(node)
        
        # 8. Вивід
        if verbose:
            status = "✓ PASS" if success else "✗ FAIL"
            corr_str = f"{correlation:.4f}"
            snr_str = f"{current_snr:.2f}"
            print(f"{node:<6} | {snr_str:<10} | {corr_str:<12} | {status:<8} | {interference_msg}")
        
        # 9. Зупинка при повній деградації
        if current_snr < -10:
            if verbose:
                print("-"*90)
                print(f"⚠️  CRITICAL: SNR dropped to {current_snr:.2f} dB. Signal lost.")
            break
    
    # Підсумки
    total_nodes = node + 1
    success_rate = correct_count / total_nodes
    total_energy = sum(node_energies)
    
    if verbose:
        print("-"*90)
        print(f"\n📊 FINAL ANALYSIS")
        print(f"   Nodes passed: {correct_count}/{total_nodes} ({success_rate*100:.1f}%)")
        print(f"   Total energy: {total_energy} aJ")
        print(f"   Avg correlation: {np.mean(correlations):.4f}")
        if errors_by_node:
            print(f"   Error nodes: {errors_by_node}")
        print("="*90 + "\n")
    
    return {
        'success_rate': success_rate,
        'correct_count': correct_count,
        'total_nodes': total_nodes,
        'total_energy': total_energy,
        'correlations': correlations,
        'snr_history': snr_history,
        'errors_by_node': errors_by_node,
        'original_bits': orig_bits,
        'frequencies': frequencies
    }


# ==================== ВІЗУАЛІЗАЦІЯ ====================

def visualize_spectral_atlas(atlas: Dict[int, np.ndarray], frequencies: np.ndarray, 
                             save_path: Optional[str] = None):
    """Візуалізує спектральний атлас"""
    plt.figure(figsize=(12, 6))
    
    for idx, profile in atlas.items():
        plt.plot(frequencies, profile, alpha=0.6, linewidth=1, label=f'Symbol {idx}')
    
    plt.xlabel('Frequency (arb. units)')
    plt.ylabel('Spectral Intensity')
    plt.title('Spectral Atlas: 16 Distinct Photonic Signatures')
    plt.legend(ncol=4, fontsize=8)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_simulation_results(results: dict):
    """Візуалізує результати симуляції"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. SNR по вузлах
    ax1 = axes[0, 0]
    ax1.plot(results['snr_history'], 'b-', linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Noise floor')
    ax1.set_xlabel('Node')
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('Signal Degradation')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Кореляція по вузлах
    ax2 = axes[0, 1]
    ax2.plot(results['correlations'], 'g-', linewidth=2)
    ax2.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Reliability threshold')
    ax2.set_xlabel('Node')
    ax2.set_ylabel('Spectral Correlation')
    ax2.set_title('Decoding Confidence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Помилки по вузлах
    ax3 = axes[1, 0]
    errors = np.zeros(results['total_nodes'])
    for node in results['errors_by_node']:
        errors[node] = 1
    ax3.bar(range(results['total_nodes']), errors, color='red', alpha=0.7)
    ax3.set_xlabel('Node')
    ax3.set_ylabel('Error (1 = FAIL)')
    ax3.set_title('Error Distribution')
    ax3.set_ylim(0, 1.2)
    ax3.grid(True, alpha=0.3)
    
    # 4. Успішність
    ax4 = axes[1, 1]
    success_rate = results['success_rate']
    ax4.pie([success_rate, 1 - success_rate], 
            labels=[f'PASS ({success_rate*100:.1f}%)', f'FAIL ({(1-success_rate)*100:.1f}%)'],
            colors=['green', 'red'], autopct='%1.1f%%')
    ax4.set_title('Overall System Performance')
    
    plt.tight_layout()
    plt.show()


# ==================== СТРЕС-ТЕСТ ====================

def run_stress_test_scenarios():
    """Запускає серію тестів з різними параметрами"""
    print("\n" + "🔥"*20)
    print(" SPECTRAL ATLAS STRESS TEST SUITE ")
    print("🔥"*20)
    
    scenarios = [
        {"name": "BASELINE", "initial_snr": 20.0, "nodes": 25, "enable_interference": False, "dispersion": False},
        {"name": "WITH INTERFERENCE", "initial_snr": 20.0, "nodes": 25, "enable_interference": True, "dispersion": False},
        {"name": "FULL PHYSICS", "initial_snr": 20.0, "nodes": 25, "enable_interference": True, "dispersion": True},
        {"name": "LOW SNR START", "initial_snr": 15.0, "nodes": 20, "enable_interference": True, "dispersion": True},
        {"name": "EXTREME STRESS", "initial_snr": 18.0, "nodes": 35, "enable_interference": True, "dispersion": True},
    ]
    
    results_summary = []
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"📡 SCENARIO: {scenario['name']}")
        print(f"   SNR start: {scenario['initial_snr']} dB | Nodes: {scenario['nodes']}")
        print(f"   Interference: {scenario['enable_interference']} | Dispersion: {scenario.get('dispersion', False)}")
        print('='*60)
        
        result = run_spectral_atlas_simulation(
            input_bits=(1, 0, 1, 0),
            initial_snr=scenario['initial_snr'],
            nodes=scenario['nodes'],
            enable_interference=scenario['enable_interference'],
            enable_dispersion=scenario.get('dispersion', False),
            verbose=True
        )
        results_summary.append({
            "scenario": scenario['name'],
            "success_rate": result['success_rate'],
            "total_energy": result['total_energy']
        })
    
    # Фінальне порівняння
    print("\n" + "📊"*20)
    print(" STRESS TEST SUMMARY ")
    print("📊"*20)
    print(f"{'Scenario':<20} | {'Success Rate':<15} | {'Total Energy (aJ)':<18}")
    print("-"*55)
    for r in results_summary:
        print(f"{r['scenario']:<20} | {r['success_rate']*100:>6.1f}%{'':<8} | {r['total_energy']:<18}")
    
    return results_summary


# ==================== MAIN ====================

if __name__ == "__main__":
    # 1. Створюємо та візуалізуємо спектральний атлас
    print("🔬 CREATING SPECTRAL ATLAS...")
    atlas, freqs, centers = create_spectral_atlas()
    visualize_spectral_atlas(atlas, freqs)
    
    # 2. Запускаємо базову симуляцію
    print("\n🚀 RUNNING BASE SIMULATION...")
    results = run_spectral_atlas_simulation(
        input_bits=(1, 0, 1, 0),
        initial_snr=20.0,
        nodes=30,
        enable_interference=True,
        enable_dispersion=True,
        verbose=True
    )
    
    # 3. Візуалізуємо результати
    visualize_simulation_results(results)
    
    # 4. Запускаємо стрес-тести
    print("\n⚡ RUNNING STRESS TEST SUITE...")
    stress_results = run_stress_test_scenarios()
    
    # 5. Фінальний аналіз
    print("\n" + "✨"*20)
    print(" SIMULATION COMPLETE ")
    print("✨"*20)
    print(f"\nSpectral Atlas performance: {results['success_rate']*100:.1f}% success over {results['total_nodes']} nodes")
    print(f"Total energy consumption: {results['total_energy']} aJ")
    print(f"Spectral correlation maintained at {np.mean(results['correlations']):.3f} average")
