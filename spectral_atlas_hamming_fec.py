import numpy as np

def run_stress_test_spectral_atlas(input_bits=(1, 0, 1, 0), initial_snr=20.0, nodes=25):
	"""
	Экстремальный тест фотонного конвейера:
	- Динамическое падение SNR (ускоренное)
	- Soft-Decision Hamming(7,4) против заградительного шума
	"""
	
	# Порождающая матрица G для Hamming(7,4) (Очищено от спецсимволов)
	G = np.array([[1, 1, 0, 1, 0, 0, 0],
				  [0, 1, 1, 0, 1, 0, 0],
				  [1, 1, 1, 0, 0, 1, 0],
				  [1, 0, 1, 0, 0, 0, 1]])

	def encode(bits):
		return np.dot(bits, G) % 2

	def soft_decode_llr(received_signal, current_snr):
		# Расчет LLR: чем ниже SNR, тем выше неопределенность
		# Защита от деления на ноль при критических SNR
		variance = 1.0 / (10**(current_snr / 10.0) + 1e-9)
		llr = (2.0 * received_signal) / variance
		
		best_score = -np.inf
		best_msg = None
		
		# Maximum Likelihood Decoding (перебор 16 комбинаций)
		for i in range(16):
			msg = np.array([int(x) for x in bin(i)[2:].zfill(4)])
			codeword = encode(msg)
			bipolar_cw = 2 * codeword - 1
			score = np.sum(llr * bipolar_cw)
			
			if score > best_score:
				best_score = score
				best_msg = msg
		return best_msg

	# Инициализация данных
	orig_bits = np.array(input_bits)
	encoded_vector = encode(orig_bits)
	bipolar_signal = 2 * encoded_vector - 1
	
	current_snr = initial_snr
	total_energy = 0
	correct_count = 0
	
	header = f"\n{'='*85}\nHARDCORE SPECTRAL ATLAS SIMULATION | START SNR: {initial_snr} dB\nTARGET DATA: {orig_bits}\n{'='*85}"
	print(header)
	print(f"{'Node':<8} | {'Status':<12} | {'SNR (dB)':<10} | {'Energy (aJ)':<12} | {'Notes'}")
	print("-" * 85)

	for n in range(nodes):
		# Ускоренное падение SNR + накопление шума
		drop = 0.7 + (n * 0.06)**1.8
		current_snr -= drop
		
		# Канал с белым шумом
		sigma = np.sqrt(1.0 / (10**(current_snr / 10.0) + 1e-9))
		noise = np.random.normal(0, sigma, len(bipolar_signal))
		
		# Burst Noise: 15% шанс резкого искажения спектра
		note = "Normal decay"
		if np.random.rand() < 0.15:
			noise += np.random.normal(0, sigma * 3.5, len(bipolar_signal))
			note = "!! BURST !!"
			
		received = bipolar_signal + noise
		
		# Мягкое декодирование (вытягиваем сигнал из-под шума)
		decoded = soft_decode_llr(received, current_snr)
		success = np.array_equal(decoded, orig_bits)
		
		# Энергопотребление узла (QD Laser + FEC logic)
		node_energy = int(84 * (1.025 ** n))
		total_energy += node_energy
		
		status = "✓" if success else "✗ FAILED"
		if success: correct_count += 1
		
		print(f"{n:<8} | {status:<12} | {current_snr:<10.2f} | {node_energy:<12} | {note}")
		
		# Предел физической выживаемости
		if current_snr < -8:
			print("-" * 85)
			print(f"CRITICAL ERROR: SNR dropped to {current_snr:.2f} dB. Signal is dead.")
			break

	print("-" * 85)
	print(f"FINAL RESULT: {correct_count}/{n+1} nodes passed | Total: {total_energy} aJ\n")

if __name__ == "__main__":
	# Запуск с варьирующимися входными данными
	test_data = np.random.randint(0, 2, 4)
	run_stress_test_spectral_atlas(input_bits=test_data, initial_snr=22.0, nodes=30)
