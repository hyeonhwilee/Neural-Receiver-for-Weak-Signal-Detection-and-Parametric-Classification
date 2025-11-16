"""
Signal Generator for IQ Data with Various Modulation Types
Generates baseband IQ samples for weak signal detection and classification
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SignalParams:
    """Parameters for generated signal"""
    center_freq: float  # Normalized frequency [-0.5, 0.5]
    bandwidth: float    # Normalized bandwidth
    power: float        # Signal power in linear scale
    snr_db: float      # Signal-to-noise ratio in dB
    symbol_rate: float  # Symbol rate (for digital signals)
    modulation_type: str


class ModulationType:
    """Modulation type constants"""
    NO_SIGNAL = 0
    AM = 1
    FM = 2
    FSK_2 = 3
    CW_RADAR = 4
    PULSED_RADAR = 5
    LFM_CHIRP = 6
    PHASE_CODED_PULSE = 7
    FREQ_CODED_PULSE = 8
    UNMODULATED_PULSE = 9
    PULSE_COMPRESSION_MOP = 10
    OTHER_MOP = 11

    @staticmethod
    def get_num_classes():
        return 12

    @staticmethod
    def get_class_names():
        return [
            'No-signal', 'AM', 'FM', '2FSK', 'CW Radar', 'Pulsed Radar',
            'LFM/Chirp', 'Phase-Coded Pulse', 'Frequency-Coded Pulse',
            'Unmodulated Pulse', 'Pulse-Compression MOP', 'Other MOP Types'
        ]

    @staticmethod
    def name_to_idx(name: str) -> int:
        names = ModulationType.get_class_names()
        return names.index(name)


class SignalGenerator:
    """Generate IQ data with various modulation types"""

    def __init__(self, sample_rate: float = 1.0, seed: Optional[int] = None):
        """
        Args:
            sample_rate: Sample rate in Hz (normalized to 1.0 for baseband)
            seed: Random seed for reproducibility
        """
        self.sample_rate = sample_rate
        if seed is not None:
            np.random.seed(seed)

    def generate_noise(self, n_samples: int, noise_power: float = 1.0) -> np.ndarray:
        """Generate complex AWGN"""
        noise = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        noise = noise / np.sqrt(2)  # Normalize to unit power
        return noise * np.sqrt(noise_power)

    def generate_am(self, n_samples: int, params: SignalParams) -> np.ndarray:
        """Generate AM modulated signal"""
        t = np.arange(n_samples) / self.sample_rate

        # Message signal (audio-like)
        fm = params.symbol_rate  # Message frequency
        message = np.sin(2 * np.pi * fm * t)

        # AM modulation
        modulation_index = 0.5
        carrier_freq = params.center_freq * self.sample_rate
        carrier = np.exp(2j * np.pi * carrier_freq * t)

        am_signal = (1 + modulation_index * message) * carrier
        am_signal = am_signal * np.sqrt(params.power)

        return am_signal

    def generate_fm(self, n_samples: int, params: SignalParams) -> np.ndarray:
        """Generate FM modulated signal"""
        t = np.arange(n_samples) / self.sample_rate

        # Message signal
        fm = params.symbol_rate
        message = np.sin(2 * np.pi * fm * t)

        # FM modulation
        freq_deviation = params.bandwidth * 0.5 * self.sample_rate
        carrier_freq = params.center_freq * self.sample_rate

        phase = 2 * np.pi * carrier_freq * t + 2 * np.pi * freq_deviation * np.cumsum(message) / self.sample_rate
        fm_signal = np.exp(1j * phase)
        fm_signal = fm_signal * np.sqrt(params.power)

        return fm_signal

    def generate_2fsk(self, n_samples: int, params: SignalParams) -> np.ndarray:
        """Generate 2FSK modulated signal"""
        t = np.arange(n_samples) / self.sample_rate

        # Generate random binary data
        symbol_duration = 1.0 / params.symbol_rate
        samples_per_symbol = max(1, int(symbol_duration * self.sample_rate))

        # Calculate number of symbols needed (with extra to ensure enough samples)
        n_symbols = int(np.ceil(n_samples / samples_per_symbol)) + 1
        binary_data = np.random.randint(0, 2, n_symbols)

        # Repeat each symbol for its duration and ensure exactly n_samples
        data_upsampled = np.repeat(binary_data, samples_per_symbol)

        # Ensure we have exactly n_samples (pad or truncate)
        if len(data_upsampled) < n_samples:
            # Pad with last value if needed
            data_upsampled = np.pad(data_upsampled, (0, n_samples - len(data_upsampled)),
                                   mode='edge')
        else:
            # Truncate if needed
            data_upsampled = data_upsampled[:n_samples]

        # FSK modulation
        freq_separation = params.bandwidth * self.sample_rate * 0.5
        f1 = params.center_freq * self.sample_rate - freq_separation / 2
        f2 = params.center_freq * self.sample_rate + freq_separation / 2

        freq = np.where(data_upsampled == 0, f1, f2)
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate

        fsk_signal = np.exp(1j * phase)
        fsk_signal = fsk_signal * np.sqrt(params.power)

        return fsk_signal

    def generate_cw_radar(self, n_samples: int, params: SignalParams) -> np.ndarray:
        """Generate CW (Continuous Wave) radar signal"""
        t = np.arange(n_samples) / self.sample_rate
        carrier_freq = params.center_freq * self.sample_rate

        cw_signal = np.exp(2j * np.pi * carrier_freq * t)
        cw_signal = cw_signal * np.sqrt(params.power)

        return cw_signal

    def generate_pulsed_radar(self, n_samples: int, params: SignalParams) -> np.ndarray:
        """Generate pulsed radar signal"""
        t = np.arange(n_samples) / self.sample_rate

        # Pulse parameters
        prf = params.symbol_rate  # Pulse Repetition Frequency
        pulse_width = 1.0 / (prf * 10)  # 10% duty cycle

        # Create pulse train
        pulse_train = np.zeros(n_samples)
        pulse_period = int(self.sample_rate / prf)
        pulse_samples = int(pulse_width * self.sample_rate)

        for i in range(0, n_samples, pulse_period):
            end_idx = min(i + pulse_samples, n_samples)
            pulse_train[i:end_idx] = 1.0

        # Modulate with carrier
        carrier_freq = params.center_freq * self.sample_rate
        carrier = np.exp(2j * np.pi * carrier_freq * t)

        pulsed_signal = pulse_train * carrier
        pulsed_signal = pulsed_signal * np.sqrt(params.power / np.mean(pulse_train**2 + 1e-10))

        return pulsed_signal

    def generate_lfm_chirp(self, n_samples: int, params: SignalParams) -> np.ndarray:
        """Generate LFM (Linear Frequency Modulation) chirp radar signal"""
        t = np.arange(n_samples) / self.sample_rate

        # Chirp parameters
        chirp_duration = 1.0 / params.symbol_rate
        bandwidth = params.bandwidth * self.sample_rate

        # Create chirp pulses
        n_chirps = int(n_samples / (chirp_duration * self.sample_rate))
        chirp_samples = int(chirp_duration * self.sample_rate)

        signal = np.zeros(n_samples, dtype=complex)

        for i in range(n_chirps):
            start_idx = int(i * chirp_duration * self.sample_rate)
            end_idx = min(start_idx + chirp_samples, n_samples)
            n_chirp = end_idx - start_idx

            t_chirp = np.arange(n_chirp) / self.sample_rate

            # LFM chirp
            chirp_rate = bandwidth / chirp_duration
            f_start = params.center_freq * self.sample_rate - bandwidth / 2
            phase = 2 * np.pi * (f_start * t_chirp + 0.5 * chirp_rate * t_chirp**2)

            signal[start_idx:end_idx] = np.exp(1j * phase)

        signal = signal * np.sqrt(params.power)

        return signal

    def generate_phase_coded_pulse(self, n_samples: int, params: SignalParams) -> np.ndarray:
        """Generate phase-coded pulse (e.g., Barker code)"""
        t = np.arange(n_samples) / self.sample_rate

        # Barker code (13-bit)
        barker_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])

        # Repeat code sequence
        chip_duration = 1.0 / (params.symbol_rate * len(barker_13))
        samples_per_chip = int(chip_duration * self.sample_rate)

        # Create phase-coded sequence
        phase_code = np.repeat(barker_13, samples_per_chip)
        n_repeats = int(np.ceil(n_samples / len(phase_code)))
        phase_code = np.tile(phase_code, n_repeats)[:n_samples]

        # Apply to carrier
        carrier_freq = params.center_freq * self.sample_rate
        carrier = np.exp(2j * np.pi * carrier_freq * t)

        # Phase shift keying
        signal = carrier * phase_code
        signal = signal * np.sqrt(params.power)

        return signal

    def generate_freq_coded_pulse(self, n_samples: int, params: SignalParams) -> np.ndarray:
        """Generate frequency-coded pulse"""
        t = np.arange(n_samples) / self.sample_rate

        # Frequency hopping pattern
        n_hops = 8
        hop_duration = 1.0 / (params.symbol_rate * n_hops)
        samples_per_hop = int(hop_duration * self.sample_rate)

        # Random frequency hops
        freq_offset = params.bandwidth * self.sample_rate * 0.4
        freq_hops = np.random.choice([-1, -0.5, 0, 0.5, 1], n_hops) * freq_offset

        # Create frequency-coded signal
        signal = np.zeros(n_samples, dtype=complex)

        for i, freq_hop in enumerate(freq_hops):
            start_idx = i * samples_per_hop
            end_idx = min(start_idx + samples_per_hop, n_samples)
            n_hop = end_idx - start_idx

            t_hop = np.arange(n_hop) / self.sample_rate
            freq = params.center_freq * self.sample_rate + freq_hop
            signal[start_idx:end_idx] = np.exp(2j * np.pi * freq * t_hop)

        signal = signal * np.sqrt(params.power)

        return signal

    def generate_unmodulated_pulse(self, n_samples: int, params: SignalParams) -> np.ndarray:
        """Generate simple unmodulated pulse"""
        t = np.arange(n_samples) / self.sample_rate

        # Single pulse or pulse train
        pulse_width = 1.0 / (params.symbol_rate * 5)
        pulse_samples = int(pulse_width * self.sample_rate)

        pulse = np.zeros(n_samples)
        pulse[:pulse_samples] = 1.0

        # Apply carrier
        carrier_freq = params.center_freq * self.sample_rate
        carrier = np.exp(2j * np.pi * carrier_freq * t)

        signal = pulse * carrier
        signal = signal * np.sqrt(params.power / np.mean(pulse**2 + 1e-10))

        return signal

    def generate_pulse_compression_mop(self, n_samples: int, params: SignalParams) -> np.ndarray:
        """Generate pulse compression MOP (combination of techniques)"""
        # Combine chirp with phase coding
        t = np.arange(n_samples) / self.sample_rate

        pulse_duration = 1.0 / params.symbol_rate
        pulse_samples = int(pulse_duration * self.sample_rate)

        signal = np.zeros(n_samples, dtype=complex)

        # Generate chirp with phase coding
        bandwidth = params.bandwidth * self.sample_rate
        chirp_rate = bandwidth / pulse_duration

        for i in range(0, n_samples, pulse_samples):
            end_idx = min(i + pulse_samples, n_samples)
            n_pulse = end_idx - i

            t_pulse = np.arange(n_pulse) / self.sample_rate

            # LFM chirp
            f_start = params.center_freq * self.sample_rate - bandwidth / 2
            phase = 2 * np.pi * (f_start * t_pulse + 0.5 * chirp_rate * t_pulse**2)

            # Add phase coding
            phase_mod = np.random.choice([0, np.pi], n_pulse)
            signal[i:end_idx] = np.exp(1j * (phase + phase_mod))

        signal = signal * np.sqrt(params.power)

        return signal

    def generate_other_mop(self, n_samples: int, params: SignalParams) -> np.ndarray:
        """Generate other MOP types (composite waveform)"""
        # Mix of different techniques
        signal1 = self.generate_lfm_chirp(n_samples, params) * 0.5
        signal2 = self.generate_freq_coded_pulse(n_samples, params) * 0.5

        signal = signal1 + signal2
        signal = signal * np.sqrt(params.power / np.mean(np.abs(signal)**2 + 1e-10))

        return signal

    def generate_signal(self, n_samples: int, params: SignalParams,
                       add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate signal with specified parameters

        Args:
            n_samples: Number of samples to generate
            params: Signal parameters
            add_noise: Whether to add AWGN

        Returns:
            signal: Clean signal (without noise)
            noisy_signal: Signal with AWGN
        """
        # Generate signal based on modulation type
        mod_generators = {
            'No-signal': lambda n, p: np.zeros(n, dtype=complex),
            'AM': self.generate_am,
            'FM': self.generate_fm,
            '2FSK': self.generate_2fsk,
            'CW Radar': self.generate_cw_radar,
            'Pulsed Radar': self.generate_pulsed_radar,
            'LFM/Chirp': self.generate_lfm_chirp,
            'Phase-Coded Pulse': self.generate_phase_coded_pulse,
            'Frequency-Coded Pulse': self.generate_freq_coded_pulse,
            'Unmodulated Pulse': self.generate_unmodulated_pulse,
            'Pulse-Compression MOP': self.generate_pulse_compression_mop,
            'Other MOP Types': self.generate_other_mop,
        }

        generator = mod_generators.get(params.modulation_type)
        if generator is None:
            raise ValueError(f"Unknown modulation type: {params.modulation_type}")

        signal = generator(n_samples, params)

        if add_noise:
            # Calculate noise power from SNR
            signal_power = np.mean(np.abs(signal)**2)
            snr_linear = 10**(params.snr_db / 10)
            noise_power = signal_power / snr_linear if snr_linear > 0 else 1.0

            noise = self.generate_noise(n_samples, noise_power)
            noisy_signal = signal + noise
        else:
            noisy_signal = signal

        return signal, noisy_signal

    def generate_random_params(self, include_no_signal: bool = True,
                              snr_range: Tuple[float, float] = (-10, 0)) -> SignalParams:
        """
        Generate random signal parameters

        Args:
            include_no_signal: Whether to include no-signal class
            snr_range: Range of SNR values in dB

        Returns:
            Random signal parameters
        """
        class_names = ModulationType.get_class_names()
        if not include_no_signal:
            class_names = class_names[1:]  # Exclude 'No-signal'

        modulation_type = np.random.choice(class_names)

        # Random parameters
        center_freq = np.random.uniform(-0.3, 0.3)  # Normalized frequency
        bandwidth = np.random.uniform(0.05, 0.2)
        snr_db = np.random.uniform(*snr_range)

        # Signal power (we'll adjust with noise)
        power = 1.0

        # Symbol rate depends on modulation type
        if 'Radar' in modulation_type or 'Pulse' in modulation_type or 'MOP' in modulation_type:
            symbol_rate = np.random.uniform(100, 1000)  # PRF for radar
        else:
            symbol_rate = np.random.uniform(100, 500)  # Symbol rate for comm signals

        if modulation_type == 'No-signal':
            power = 0.0
            snr_db = -np.inf

        return SignalParams(
            center_freq=center_freq,
            bandwidth=bandwidth,
            power=power,
            snr_db=snr_db,
            symbol_rate=symbol_rate,
            modulation_type=modulation_type
        )
