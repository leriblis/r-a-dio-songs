"""
Audio feature extraction & DJ set analysis for r/a/dio.

Extracts per-track features (BPM, key, loudness, energy curve, spectral),
computes transition quality between consecutive tracks, and produces
set-level metrics for comparing DJ mixing quality.

Usage:
    uv run audio_analysis.py analyze <set_dir>      # Analyze one DJ set
    uv run audio_analysis.py compare <dir1> <dir2>   # Compare two sets
    uv run audio_analysis.py report <set_dir>         # Full narrative report
"""

import json
import os
import glob
import argparse
import warnings
from collections import defaultdict

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

PWD = os.path.dirname(os.path.realpath(__file__))


# ─── Camelot Wheel for harmonic compatibility ──────────────────────────
# Maps (key, mode) -> Camelot notation
# mode: 'major' or 'minor'
CAMELOT = {
    ('C', 'major'): '8B', ('A', 'minor'): '8A',
    ('G', 'major'): '9B', ('E', 'minor'): '9A',
    ('D', 'major'): '10B', ('B', 'minor'): '10A',
    ('A', 'major'): '11B', ('F#', 'minor'): '11A',
    ('E', 'major'): '12B', ('C#', 'minor'): '12A',
    ('B', 'major'): '1B', ('G#', 'minor'): '1A',
    ('F#', 'major'): '2B', ('D#', 'minor'): '2A',
    ('Db', 'major'): '2B', ('Bb', 'minor'): '2A',
    ('Ab', 'major'): '3B', ('F', 'minor'): '3A',
    ('Eb', 'major'): '4B', ('C', 'minor'): '4A',
    ('Bb', 'major'): '5B', ('G', 'minor'): '5A',
    ('F', 'major'): '6B', ('D', 'minor'): '6A',
    # enharmonic
    ('Gb', 'major'): '2B', ('Eb', 'minor'): '2A',
}

# Reverse: Camelot code -> number, letter
def _parse_camelot(code):
    if not code or len(code) < 2:
        return None, None
    letter = code[-1]
    try:
        num = int(code[:-1])
    except ValueError:
        return None, None
    return num, letter

def camelot_distance(cam_a, cam_b):
    """Compute harmonic distance on Camelot wheel. 0 = perfect, lower = better."""
    if not cam_a or not cam_b:
        return 7  # unknown = neutral
    na, la = _parse_camelot(cam_a)
    nb, lb = _parse_camelot(cam_b)
    if na is None or nb is None:
        return 7
    # Same position
    if na == nb and la == lb:
        return 0
    # Same number, different letter (relative major/minor)
    if na == nb:
        return 1
    # Adjacent number, same letter
    diff = min(abs(na - nb), 12 - abs(na - nb))
    if la == lb and diff == 1:
        return 1
    # Two steps away
    if la == lb and diff == 2:
        return 2
    return min(diff + (0 if la == lb else 1), 7)


# ─── Feature Extraction ───────────────────────────────────────────────

def extract_features(filepath, sr=22050):
    """
    Extract audio features from a single track.

    Returns dict with:
        - bpm: float
        - key: str (e.g. 'C major')
        - camelot: str (e.g. '8B')
        - loudness_lufs: float
        - rms_mean: float
        - rms_std: float
        - rms_curve: list[float] (downsampled to ~1 value per 2 seconds)
        - spectral_centroid_mean: float
        - spectral_centroid_std: float
        - mfcc_mean: list[float] (13 coefficients)
        - onset_rate: float (onsets per second)
        - duration_sec: float
        - spectral_rolloff_mean: float
        - zero_crossing_rate_mean: float
    """
    try:
        y, sr_actual = librosa.load(filepath, sr=sr, mono=True)
    except Exception as e:
        return {'error': str(e), 'filename': os.path.basename(filepath)}

    duration = len(y) / sr_actual
    if duration < 10:
        return {'error': 'too_short', 'duration_sec': duration, 'filename': os.path.basename(filepath)}

    features = {
        'filename': os.path.basename(filepath),
        'duration_sec': round(duration, 2),
    }

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr_actual)
    # librosa returns array in newer versions
    bpm = float(np.atleast_1d(tempo)[0])
    features['bpm'] = round(bpm, 1)

    # Key detection via chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr_actual)
    chroma_mean = chroma.mean(axis=1)

    # Krumhansl-Kessler profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    best_corr = -2
    best_key = 'C'
    best_mode = 'major'

    for i in range(12):
        shifted = np.roll(chroma_mean, -i)
        corr_maj = np.corrcoef(shifted, major_profile)[0, 1]
        corr_min = np.corrcoef(shifted, minor_profile)[0, 1]
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = notes[i]
            best_mode = 'major'
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = notes[i]
            best_mode = 'minor'

    features['key'] = f"{best_key} {best_mode}"
    features['camelot'] = CAMELOT.get((best_key, best_mode), '?')
    features['key_confidence'] = round(float(best_corr), 3)

    # Loudness (LUFS)
    try:
        # pyloudnorm needs audio at original sample rate for accurate LUFS
        y_loud, sr_loud = sf.read(filepath)
        if y_loud.ndim > 1:
            y_loud = y_loud.mean(axis=1)
        meter = pyloudnorm.Meter(sr_loud)
        loudness = meter.integrated_loudness(y_loud)
        features['loudness_lufs'] = round(float(loudness), 2) if np.isfinite(loudness) else -70.0
    except Exception:
        features['loudness_lufs'] = -70.0

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    features['rms_mean'] = round(float(rms.mean()), 6)
    features['rms_std'] = round(float(rms.std()), 6)
    # Downsample RMS curve to ~1 point per 2 seconds
    frames_per_2sec = max(1, int(2 * sr_actual / 512))
    rms_downsampled = []
    for i in range(0, len(rms), frames_per_2sec):
        chunk = rms[i:i+frames_per_2sec]
        rms_downsampled.append(round(float(chunk.mean()), 6))
    features['rms_curve'] = rms_downsampled

    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr_actual)[0]
    features['spectral_centroid_mean'] = round(float(centroid.mean()), 2)
    features['spectral_centroid_std'] = round(float(centroid.std()), 2)

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr_actual)[0]
    features['spectral_rolloff_mean'] = round(float(rolloff.mean()), 2)

    # Zero crossing rate (noise/percussion indicator)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zero_crossing_rate_mean'] = round(float(zcr.mean()), 6)

    # MFCCs (timbre fingerprint)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_actual, n_mfcc=13)
    features['mfcc_mean'] = [round(float(x), 3) for x in mfcc.mean(axis=1)]

    # Onset rate (rhythmic density)
    onsets = librosa.onset.onset_detect(y=y, sr=sr_actual, units='time')
    features['onset_rate'] = round(len(onsets) / duration, 3) if duration > 0 else 0.0

    # Energy percentiles for dynamics analysis
    np.sort(rms)
    features['rms_p10'] = round(float(np.percentile(rms, 10)), 6)
    features['rms_p50'] = round(float(np.percentile(rms, 50)), 6)
    features['rms_p90'] = round(float(np.percentile(rms, 90)), 6)
    features['dynamic_range'] = round(features['rms_p90'] - features['rms_p10'], 6)

    return features


def extract_transition_features(feat_a, feat_b):
    """
    Compute transition quality between two consecutive tracks.

    Returns dict with transition metrics.
    """
    if 'error' in feat_a or 'error' in feat_b:
        return {'quality': 'unknown', 'error': 'missing_features'}

    trans = {}

    # BPM compatibility
    bpm_a, bpm_b = feat_a['bpm'], feat_b['bpm']
    # Check for tempo ratios (1:1, 1:2, 2:1)
    if bpm_a > 0 and bpm_b > 0:
        ratio = bpm_b / bpm_a
        # Find closest "natural" ratio
        best_ratio_dist = min(
            abs(ratio - 1.0),
            abs(ratio - 0.5),
            abs(ratio - 2.0),
            abs(ratio - 0.75),
            abs(ratio - 1.5),
        )
        trans['bpm_ratio'] = round(ratio, 3)
        trans['bpm_delta'] = round(abs(bpm_b - bpm_a), 1)
        trans['bpm_compat'] = round(max(0, 1 - best_ratio_dist * 2), 3)
    else:
        trans['bpm_compat'] = 0.5

    # Key compatibility (Camelot distance)
    cam_dist = camelot_distance(feat_a.get('camelot'), feat_b.get('camelot'))
    trans['camelot_distance'] = cam_dist
    trans['key_compat'] = round(max(0, 1 - cam_dist / 7), 3)

    # Energy continuity (RMS delta)
    rms_delta = abs(feat_b['rms_mean'] - feat_a['rms_mean'])
    max_rms = max(feat_a['rms_mean'], feat_b['rms_mean'], 0.001)
    trans['energy_delta'] = round(rms_delta / max_rms, 3)
    trans['energy_compat'] = round(max(0, 1 - trans['energy_delta']), 3)

    # Loudness continuity
    lufs_delta = abs(feat_b['loudness_lufs'] - feat_a['loudness_lufs'])
    trans['loudness_delta'] = round(lufs_delta, 2)
    trans['loudness_compat'] = round(max(0, 1 - lufs_delta / 12), 3)

    # Timbral continuity (MFCC cosine similarity)
    mfcc_a = np.array(feat_a['mfcc_mean'])
    mfcc_b = np.array(feat_b['mfcc_mean'])
    cos_sim = np.dot(mfcc_a, mfcc_b) / (np.linalg.norm(mfcc_a) * np.linalg.norm(mfcc_b) + 1e-9)
    trans['timbre_similarity'] = round(float(cos_sim), 3)

    # Spectral continuity
    centroid_delta = abs(feat_b['spectral_centroid_mean'] - feat_a['spectral_centroid_mean'])
    max_centroid = max(feat_a['spectral_centroid_mean'], feat_b['spectral_centroid_mean'], 1)
    trans['spectral_delta'] = round(centroid_delta / max_centroid, 3)

    # Overall transition quality (weighted composite)
    trans['quality_score'] = round(
        trans['bpm_compat'] * 0.25 +
        trans['key_compat'] * 0.20 +
        trans['energy_compat'] * 0.20 +
        trans['loudness_compat'] * 0.15 +
        trans['timbre_similarity'] * 0.10 +
        max(0, 1 - trans['spectral_delta']) * 0.10,
        3
    )

    return trans


# ─── Set-Level Analysis ───────────────────────────────────────────────

def analyze_set(set_dir):
    """
    Analyze a full DJ set: extract features for all tracks, compute transitions,
    and produce set-level metrics.
    """
    # Find all audio files, sorted by track number
    patterns = ['*.mp3', '*.flac', '*.ogg', '*.wav', '*.m4a', '*.opus']
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(set_dir, pat)))

    if not files:
        print(f"No audio files found in {set_dir}")
        return None

    # Sort by filename (should be numbered: 001., 002., etc.)
    files.sort(key=lambda f: os.path.basename(f))

    print(f"\nAnalyzing {len(files)} tracks from: {os.path.basename(set_dir)}")
    print("=" * 70)

    # Extract features per track
    track_features = []
    for i, filepath in enumerate(files):
        fname = os.path.basename(filepath)
        print(f"  [{i+1:>3}/{len(files)}] {fname[:65]}...", end='', flush=True)
        features = extract_features(filepath)
        track_features.append(features)
        if 'error' in features:
            print(f" ERROR: {features['error']}")
        else:
            print(f" BPM={features['bpm']:.0f} Key={features['key']} "
                  f"LUFS={features['loudness_lufs']:.1f}")

    # Compute transitions
    transitions = []
    for i in range(len(track_features) - 1):
        trans = extract_transition_features(track_features[i], track_features[i+1])
        trans['from_track'] = track_features[i].get('filename', '?')
        trans['to_track'] = track_features[i+1].get('filename', '?')
        transitions.append(trans)

    # Set-level metrics
    valid_features = [f for f in track_features if 'error' not in f]
    set_metrics = compute_set_metrics(valid_features, transitions)

    result = {
        'set_dir': set_dir,
        'dj': _extract_dj_from_path(set_dir),
        'n_tracks': len(files),
        'n_valid': len(valid_features),
        'track_features': track_features,
        'transitions': transitions,
        'set_metrics': set_metrics,
    }

    return result


def compute_set_metrics(features, transitions):
    """Compute set-level aggregate metrics from track features and transitions."""
    if len(features) < 3:
        return {'error': 'too_few_tracks'}

    metrics = {}

    # BPM analysis
    bpms = [f['bpm'] for f in features]
    metrics['bpm_mean'] = round(float(np.mean(bpms)), 1)
    metrics['bpm_std'] = round(float(np.std(bpms)), 1)
    metrics['bpm_range'] = round(max(bpms) - min(bpms), 1)
    metrics['bpm_cv'] = round(float(np.std(bpms) / np.mean(bpms)), 3) if np.mean(bpms) > 0 else 0

    # BPM flow (how BPM changes through the set)
    bpm_deltas = [abs(bpms[i+1] - bpms[i]) for i in range(len(bpms)-1)]
    metrics['avg_bpm_jump'] = round(float(np.mean(bpm_deltas)), 1)
    metrics['max_bpm_jump'] = round(float(np.max(bpm_deltas)), 1)

    # Energy arc
    rms_values = [f['rms_mean'] for f in features]
    metrics['energy_mean'] = round(float(np.mean(rms_values)), 6)
    metrics['energy_std'] = round(float(np.std(rms_values)), 6)

    # Energy arc shape: split into quarters and compare
    n = len(rms_values)
    q_size = max(1, n // 4)
    quarters = [
        np.mean(rms_values[:q_size]),
        np.mean(rms_values[q_size:2*q_size]),
        np.mean(rms_values[2*q_size:3*q_size]),
        np.mean(rms_values[3*q_size:]),
    ]
    metrics['energy_quarters'] = [round(float(q), 6) for q in quarters]

    # Does it have a peak? Where?
    # Sliding window peak detection
    window = max(3, n // 8)
    windowed_energy = []
    for i in range(len(rms_values) - window + 1):
        windowed_energy.append(np.mean(rms_values[i:i+window]))
    if windowed_energy:
        peak_idx = int(np.argmax(windowed_energy))
        peak_pos = peak_idx / max(len(windowed_energy) - 1, 1)
        metrics['energy_peak_position'] = round(peak_pos, 3)  # 0=start, 1=end
        metrics['energy_peak_value'] = round(float(max(windowed_energy)), 6)
        valley_idx = int(np.argmin(windowed_energy))
        valley_pos = valley_idx / max(len(windowed_energy) - 1, 1)
        metrics['energy_valley_position'] = round(valley_pos, 3)

    # Direction changes (ebb and flow)
    direction_changes = 0
    for i in range(2, len(windowed_energy)):
        if (windowed_energy[i] - windowed_energy[i-1]) * (windowed_energy[i-1] - windowed_energy[i-2]) < 0:
            direction_changes += 1
    metrics['energy_direction_changes'] = direction_changes
    metrics['energy_wave_frequency'] = round(direction_changes / max(len(features) / 10, 1), 2)

    # Key distribution
    keys = [f['key'] for f in features]
    key_counts = defaultdict(int)
    for k in keys:
        key_counts[k] += 1
    # Most common key
    sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
    metrics['dominant_key'] = sorted_keys[0][0] if sorted_keys else '?'
    metrics['key_diversity'] = round(len(key_counts) / len(features), 3)
    metrics['top_keys'] = sorted_keys[:5]

    # Loudness analysis
    lufs_values = [f['loudness_lufs'] for f in features]
    metrics['loudness_mean'] = round(float(np.mean(lufs_values)), 2)
    metrics['loudness_std'] = round(float(np.std(lufs_values)), 2)
    metrics['loudness_range'] = round(float(np.max(lufs_values) - np.min(lufs_values)), 2)

    # Transition quality
    if transitions:
        valid_trans = [t for t in transitions if 'quality_score' in t]
        if valid_trans:
            quality_scores = [t['quality_score'] for t in valid_trans]
            metrics['avg_transition_quality'] = round(float(np.mean(quality_scores)), 3)
            metrics['min_transition_quality'] = round(float(np.min(quality_scores)), 3)
            metrics['max_transition_quality'] = round(float(np.max(quality_scores)), 3)

            # How many "smooth" transitions (quality > 0.6)?
            smooth = sum(1 for q in quality_scores if q > 0.6)
            metrics['smooth_transition_pct'] = round(smooth / len(quality_scores) * 100, 1)

            # How many "jarring" transitions (quality < 0.3)?
            jarring = sum(1 for q in quality_scores if q < 0.3)
            metrics['jarring_transition_pct'] = round(jarring / len(quality_scores) * 100, 1)

            # BPM compatibility across transitions
            bpm_compats = [t.get('bpm_compat', 0.5) for t in valid_trans]
            metrics['avg_bpm_compat'] = round(float(np.mean(bpm_compats)), 3)

            # Key compatibility
            key_compats = [t.get('key_compat', 0.5) for t in valid_trans]
            metrics['avg_key_compat'] = round(float(np.mean(key_compats)), 3)

            # Timbre continuity
            timbre_sims = [t.get('timbre_similarity', 0.5) for t in valid_trans]
            metrics['avg_timbre_continuity'] = round(float(np.mean(timbre_sims)), 3)

    # Dynamic range (track-level dynamics)
    dynamic_ranges = [f.get('dynamic_range', 0) for f in features]
    metrics['avg_dynamic_range'] = round(float(np.mean(dynamic_ranges)), 6)

    # Spectral variety
    centroids = [f['spectral_centroid_mean'] for f in features]
    metrics['spectral_variety'] = round(float(np.std(centroids) / np.mean(centroids)), 3) if np.mean(centroids) > 0 else 0

    # Total duration
    total_dur = sum(f['duration_sec'] for f in features)
    metrics['total_duration_min'] = round(total_dur / 60, 1)

    return metrics


def _extract_dj_from_path(path):
    """Extract DJ name from directory path like '2023-12-27, 1814, sauce'."""
    dirname = os.path.basename(path.rstrip('/'))
    parts = dirname.split(', ')
    if len(parts) >= 3:
        return parts[2]
    return dirname


# ─── Reporting ─────────────────────────────────────────────────────────

def print_report(result):
    """Print a human-readable analysis report."""
    if not result:
        return

    m = result['set_metrics']
    if 'error' in m:
        print(f"Error: {m['error']}")
        return

    dj = result['dj']
    print(f"\n{'='*80}")
    print(f"  AUDIO ANALYSIS REPORT: {dj}")
    print(f"  {result['n_tracks']} tracks ({result['n_valid']} analyzed), "
          f"{m.get('total_duration_min', 0):.0f} minutes")
    print(f"{'='*80}")

    # Tempo
    print("\n  TEMPO")
    print(f"    Average BPM: {m['bpm_mean']:.0f} (std: {m['bpm_std']:.0f})")
    print(f"    BPM range: {m['bpm_range']:.0f}")
    print(f"    Avg BPM jump between tracks: {m['avg_bpm_jump']:.1f}")
    print(f"    Max BPM jump: {m['max_bpm_jump']:.0f}")

    # Key
    print("\n  HARMONY")
    print(f"    Dominant key: {m['dominant_key']}")
    print(f"    Key diversity: {m['key_diversity']:.1%} unique keys")
    print(f"    Top keys: {', '.join(f'{k}({c})' for k,c in m.get('top_keys', []))}")

    # Energy
    print("\n  ENERGY ARC")
    eq = m.get('energy_quarters', [0,0,0,0])
    labels = ['Opening', 'Build', 'Peak', 'Closing']
    max_e = max(eq) if eq else 1
    for i, (label, val) in enumerate(zip(labels, eq)):
        bar_len = int(val / max_e * 40) if max_e > 0 else 0
        bar = '#' * bar_len
        print(f"    {label:>8}: {bar} ({val:.4f})")
    peak_pos = m.get('energy_peak_position', 0.5)
    print(f"    Peak at: {peak_pos:.0%} through set")
    print(f"    Direction changes: {m.get('energy_direction_changes', 0)} "
          f"(wave freq: {m.get('energy_wave_frequency', 0):.1f})")

    # Transitions
    print("\n  TRANSITIONS")
    print(f"    Avg quality: {m.get('avg_transition_quality', 0):.3f}")
    print(f"    Smooth (>0.6): {m.get('smooth_transition_pct', 0):.0f}%")
    print(f"    Jarring (<0.3): {m.get('jarring_transition_pct', 0):.0f}%")
    print(f"    Avg BPM compat: {m.get('avg_bpm_compat', 0):.3f}")
    print(f"    Avg key compat: {m.get('avg_key_compat', 0):.3f}")
    print(f"    Avg timbre continuity: {m.get('avg_timbre_continuity', 0):.3f}")

    # Loudness
    print("\n  LOUDNESS")
    print(f"    Average LUFS: {m['loudness_mean']:.1f}")
    print(f"    LUFS std: {m['loudness_std']:.1f}")
    print(f"    LUFS range: {m['loudness_range']:.1f}")

    # Dynamics
    print("\n  DYNAMICS & SPECTRAL")
    print(f"    Avg dynamic range: {m['avg_dynamic_range']:.4f}")
    print(f"    Spectral variety: {m.get('spectral_variety', 0):.3f}")

    # Top/worst transitions
    trans = result['transitions']
    valid_trans = [t for t in trans if 'quality_score' in t]
    if valid_trans:
        sorted_trans = sorted(valid_trans, key=lambda t: t['quality_score'], reverse=True)
        print("\n  BEST TRANSITIONS")
        for t in sorted_trans[:5]:
            print(f"    {t['quality_score']:.3f}  {t['from_track'][:35]} -> {t['to_track'][:35]}")
        print("\n  WORST TRANSITIONS")
        for t in sorted_trans[-5:]:
            print(f"    {t['quality_score']:.3f}  {t['from_track'][:35]} -> {t['to_track'][:35]}")


def compare_sets(results):
    """Compare multiple DJ set analysis results."""
    print(f"\n{'='*100}")
    print(f"  COMPARATIVE ANALYSIS: {len(results)} SETS")
    print(f"{'='*100}")

    # Table header
    djs = [r['dj'] for r in results]
    col_w = 18
    header = f"{'Metric':<35}" + "".join(f"{dj:>{col_w}}" for dj in djs)
    print(f"\n{header}")
    print("-" * (35 + col_w * len(djs)))

    metrics_to_compare = [
        ('Tracks', lambda r: r['n_valid']),
        ('Duration (min)', lambda r: r['set_metrics'].get('total_duration_min', 0)),
        ('Avg BPM', lambda r: r['set_metrics'].get('bpm_mean', 0)),
        ('BPM std', lambda r: r['set_metrics'].get('bpm_std', 0)),
        ('Avg BPM jump', lambda r: r['set_metrics'].get('avg_bpm_jump', 0)),
        ('Dominant key', lambda r: r['set_metrics'].get('dominant_key', '?')),
        ('Key diversity', lambda r: r['set_metrics'].get('key_diversity', 0)),
        ('Energy peak @', lambda r: r['set_metrics'].get('energy_peak_position', 0)),
        ('Wave frequency', lambda r: r['set_metrics'].get('energy_wave_frequency', 0)),
        ('Avg transition Q', lambda r: r['set_metrics'].get('avg_transition_quality', 0)),
        ('Smooth trans %', lambda r: r['set_metrics'].get('smooth_transition_pct', 0)),
        ('Jarring trans %', lambda r: r['set_metrics'].get('jarring_transition_pct', 0)),
        ('Avg BPM compat', lambda r: r['set_metrics'].get('avg_bpm_compat', 0)),
        ('Avg key compat', lambda r: r['set_metrics'].get('avg_key_compat', 0)),
        ('Avg timbre cont.', lambda r: r['set_metrics'].get('avg_timbre_continuity', 0)),
        ('Loudness (LUFS)', lambda r: r['set_metrics'].get('loudness_mean', 0)),
        ('Loudness range', lambda r: r['set_metrics'].get('loudness_range', 0)),
        ('Dynamic range', lambda r: r['set_metrics'].get('avg_dynamic_range', 0)),
        ('Spectral variety', lambda r: r['set_metrics'].get('spectral_variety', 0)),
    ]

    for name, getter in metrics_to_compare:
        vals = []
        for r in results:
            try:
                v = getter(r)
                vals.append(v)
            except Exception:
                vals.append('?')

        row = f"  {name:<33}"
        for v in vals:
            if isinstance(v, float):
                row += f"{v:>{col_w}.3f}"
            elif isinstance(v, int):
                row += f"{v:>{col_w}}"
            else:
                row += f"{str(v):>{col_w}}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Audio feature extraction for DJ set analysis")
    parser.add_argument('command', choices=['analyze', 'compare', 'report'],
                       help='analyze: extract features | compare: compare sets | report: full report')
    parser.add_argument('dirs', nargs='+', help='Set directory/directories to analyze')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Save results to JSON file')
    args = parser.parse_args()

    if args.command == 'analyze':
        result = analyze_set(args.dirs[0])
        if result:
            print_report(result)
            # Always cache to .audio_features.json in the set directory
            set_dir = args.dirs[0]
            cache_path = os.path.join(set_dir, '.audio_features.json')
            if os.path.isdir(set_dir):
                with open(cache_path, 'w') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=str)
                print(f"\nCached to {cache_path}")
            if args.output:
                save_result = {k: v for k, v in result.items()}
                with open(args.output, 'w') as f:
                    json.dump(save_result, f, ensure_ascii=False, indent=2, default=str)
                print(f"Saved to {args.output}")

    elif args.command == 'compare':
        results = []
        for d in args.dirs:
            # Check for cached results
            cache_path = os.path.join(d, '.audio_features.json')
            if os.path.exists(cache_path):
                print(f"Loading cached results from {cache_path}")
                with open(cache_path) as f:
                    results.append(json.load(f))
            else:
                result = analyze_set(d)
                if result:
                    results.append(result)
                    # Cache results
                    with open(cache_path, 'w') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        if len(results) >= 2:
            compare_sets(results)
        else:
            print("Need at least 2 sets to compare")

    elif args.command == 'report':
        result = analyze_set(args.dirs[0])
        if result:
            print_report(result)


if __name__ == '__main__':
    main()
