import torch
import numpy as np
import librosa
import torchaudio
from huggingface_hub import hf_hub_download
from audiocraft.metrics import CLAPTextConsistencyMetric
import bittensor as bt

class MetricEvaluator:
    @staticmethod
    def calculate_snr(file_path, silence_threshold=1e-4, constant_signal_threshold=1e-2):
        audio_signal, _ = librosa.load(file_path, sr=None)
        if np.max(np.abs(audio_signal)) < silence_threshold or np.var(audio_signal) < constant_signal_threshold:
            return 0
        signal_power = np.mean(audio_signal**2)
        noise_signal = librosa.effects.preemphasis(audio_signal)
        noise_power = np.mean(noise_signal**2)
        if noise_power < 1e-10:
            return 0
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    @staticmethod
    def calculate_hnr(file_path):
        y, _ = librosa.load(file_path, sr=None)
        if np.max(np.abs(y)) < 1e-4 or np.var(y) < 1e-2:
            return 0
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_power = np.mean(harmonic**2)
        noise_power = np.mean(percussive**2)
        hnr = 10 * np.log10(harmonic_power / max(noise_power, 1e-10))
        return hnr

    @staticmethod
    def calculate_consistency(file_path, text):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pt_file = hf_hub_download(repo_id="lukewys/laion_clap", filename="music_audioset_epoch_15_esc_90.14.pt")
            clap_metric = CLAPTextConsistencyMetric(pt_file, model_arch='HTSAT-base').to(device)

            def convert_audio(audio, from_rate, to_rate, to_channels):
                resampler = torchaudio.transforms.Resample(orig_freq=from_rate, new_freq=to_rate)
                audio = resampler(audio)
                if to_channels == 1:
                    audio = audio.mean(dim=0, keepdim=True)
                return audio

            audio, sr = torchaudio.load(file_path)
            audio = convert_audio(audio, from_rate=sr, to_rate=sr, to_channels=1)

            clap_metric.update(audio.unsqueeze(0), [text], torch.tensor([audio.shape[1]]), torch.tensor([sr]))
            consistency_score = clap_metric.compute()
            return consistency_score
        except Exception as e:
            print(f"An error occurred while calculating music consistency score: {e}")
            return None

class Normalizer:
    @staticmethod
    def normalize_quality(quality_metric):
        return 1 / (1 + np.exp(-((quality_metric - 20) / 10)))

    @staticmethod
    def normalize_consistency(score):
        if score is not None:
            if score > 0:
                normalized_consistency = (score + 1) / 2
            else:
                normalized_consistency = 0
        else:
            normalized_consistency = 0
        return normalized_consistency

class Aggregator:
    @staticmethod
    def weighted_average(scores, weights):
        weighted_sum = sum(scores[key] * weights[key] for key in scores)
        total_weight = sum(weights.values())
        return weighted_sum / total_weight

class MusicQualityEvaluator:
    def __init__(self):
        self.metric_evaluator = MetricEvaluator()
        self.normalizer = Normalizer()
        self.aggregator = Aggregator()

    def evaluate_music_quality(self, file_path, text=None):
        try:
            snr_score = self.metric_evaluator.calculate_snr(file_path)
            bt.logging.info(f'.......SNR......: {snr_score} dB')
        except:
            pass
            bt.logging.error(f"Failed to calculate SNR")

        try:
            hnr_score = self.metric_evaluator.calculate_hnr(file_path)
            bt.logging.info(f'.......HNR......: {hnr_score} dB')
        except:
            pass
            bt.logging.error(f"Failed to calculate HNR")

        try:
            consistency_score = self.metric_evaluator.calculate_consistency(file_path, text)
            bt.logging.info(f'....... Consistency Score ......: {consistency_score}')
        except:
            pass
            bt.logging.error(f"Failed to calculate Consistency score")

        # Normalize scores
        normalized_snr = self.normalizer.normalize_quality(snr_score)
        normalized_hnr = self.normalizer.normalize_quality(hnr_score)
        normalized_consistency = self.normalizer.normalize_consistency(consistency_score)

        bt.logging.info(f'Normalized Metrics: SNR = {normalized_snr}, HNR = {normalized_hnr}, Consistency = {normalized_consistency}')

        # Calculate weighted average
        scores = {
            'snr': normalized_snr,
            'hnr': normalized_hnr,
            'consistency': normalized_consistency
        }
        weights = {
            'snr': 15,
            'hnr': 15,
            'consistency': 70
        }
        aggregate_score = self.aggregator.weighted_average(scores, weights)
        bt.logging.info(f'....... Aggregate Weighted Score ......: {aggregate_score}')
        return aggregate_score
