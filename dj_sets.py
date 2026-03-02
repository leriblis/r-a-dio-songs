"""
DJ set reconstruction and scoring for r/a/dio.

Reconstructs DJ sessions from songs_db.json and scores them on
multiple independent dimensions.

Scoring dimensions:
  1. arc_shape      — Does energy ebb and flow? (NOT flat)
  2. block_coherence — Artist/franchise clustering within segments
  3. diversity      — Range of artists and genres
  4. pacing         — Songs/hour consistency, track duration variety
  5. novelty        — Deep cuts vs overplayed songs
  6. length_bonus   — Longer sets (2-6h sweet spot) get a bonus
"""

import json
import os
import sys
from datetime import datetime
from collections import Counter, defaultdict
import statistics
import argparse
import math
import re

PWD = os.path.dirname(os.path.realpath(__file__))
DBNAME = os.path.join(PWD, "songs_db.json")
RADIO_DATE_FMT = "%Y-%m-%dT%H:%M:%S%z"

# Maximum gap (minutes) between consecutive plays to consider them
# part of the same DJ session. DJs sometimes pause for talking, bathroom
# breaks, or technical issues — 15 min is generous but realistic.
SESSION_GAP_MINUTES = 15

# Minimum number of songs for a set to be scored.
# Below this, it's a cameo/test, not a real set.
MIN_SET_SONGS = 15


def load_db():
    """Load songs_db.json, return (songs_dic, dj_dic)."""
    with open(DBNAME) as f:
        data = json.load(f)
    return data["songs_dic"], data.get("dj_dic", {})


def extract_artist(song_title):
    """Extract artist from 'Artist - Song Title' format."""
    if " - " in song_title:
        return song_title.split(" - ", 1)[0].strip()
    return song_title.strip()


def split_song(song_title):
    """Split song into (artist, title) from 'Artist - Title' format."""
    if " - " in song_title:
        artist, title = song_title.split(" - ", 1)
        return artist.strip(), title.strip()
    s = song_title.strip()
    return s, s


def normalize_artist(artist):
    """Normalize artist strings to collapse small naming variants."""
    a = artist.lower()
    a = re.sub(r"\b(feat|ft|featuring|with|vs)\b.*$", "", a).strip()
    a = re.sub(r"[\(\)\[\]\{\}]", " ", a)
    a = re.sub(r"[^a-z0-9\s]", " ", a)
    a = re.sub(r"\s+", " ", a).strip()
    return a or artist.lower().strip()


THEME_STOPWORDS = {
    "the",
    "and",
    "for",
    "from",
    "with",
    "ver",
    "version",
    "tv",
    "size",
    "full",
    "short",
    "long",
    "mix",
    "edit",
    "remaster",
    "radio",
    "anime",
    "song",
    "track",
    "ost",
    "bgm",
    "op",
    "ed",
    "opening",
    "ending",
    "insert",
    "theme",
    "character",
    "vocal",
    "instrumental",
    "animever",
}

# Music/anime markers used for theme extraction and energy hints.
THEME_MARKERS = {
    "op",
    "opening",
    "ed",
    "ending",
    "ost",
    "bgm",
    "insert",
    "character song",
    "anison",
    "tv size",
    "arrange",
    "remix",
    "live",
    "acoustic",
    "piano",
    "instrumental",
    "cover",
    "ver",
}

ENERGY_HINTS = {
    "op": 1.0,
    "opening": 1.0,
    "insert": 0.7,
    "live": 0.6,
    "remix": 0.8,
    "bootleg": 0.9,
    "arrange": 0.4,
    "hardcore": 1.5,
    "hardstyle": 1.3,
    "gabber": 1.5,
    "speedcore": 1.8,
    "trance": 0.9,
    "eurobeat": 1.2,
    "ed": -0.3,
    "ending": -0.3,
    "ost": -0.4,
    "bgm": -0.5,
    "acoustic": -0.8,
    "piano": -0.8,
    "ballad": -0.7,
    "instrumental": -0.3,
    "lullaby": -1.0,
    "chill": -0.6,
}

# Keywords indicating the DJ is playing bootleg/remix versions
# rather than original tracks — a sign of deep scene knowledge.
REMIX_KEYWORDS = [
    "remix",
    "bootleg",
    "edit",
    "flip",
    "rework",
    "mashup",
    "rmx",
    "bootleg)",
    "bootleg]",
    "btlg",
    "refix",
    "remake",
]

# Electronic music scene artists that indicate doujin/underground knowledge.
SCENE_ARTISTS_LOWER = {
    "dj sharpnel",
    "sharpnel",
    "lolistyle gabbers",
    "massive new krew",
    "nhato",
    "t+pazolite",
    "camellia",
    "かめりあ",
    "usao",
    "redalice",
    "hommarju",
    "dj noriken",
    "m-project",
    "sixstylez",
    "helblinde",
    "round wave crusher",
    "laser imouto",
    "yooh",
    "xi",
    "aran",
    "getty",
    "p*light",
    "kobaryo",
    "goreshit",
    "ryu☆",
    "dj genki",
    "noma",
    "dj技術",
    "assertive",
    "sky_delta",
    "banvox",
    "cYsmix",
    "cysmix",
    "aethral",
    "alicemetix",
    "mtell",
    "srav3r",
    "da tweekaz",
    "headhunterz",
    "s3rl",
    "technikore",
    "dj cherry clone",
    "pocotan",
    "mameyudoufu",
}


def _normalize_text(s):
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[_/|]+", " ", s)
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"[^a-z0-9\s:+!-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_theme_tags(song_title):
    """
    Extract franchise/thematic tags from artist-title text.
    Returns set[str], e.g. {'fr:symphogear', 'mk:op'}.
    """
    artist, title = split_song(song_title)
    text = _normalize_text(f"{artist} {title}")
    tags = set()

    # Marker tags.
    for marker in THEME_MARKERS:
        if marker in text:
            tags.add(f"mk:{marker}")

    # Capture "<franchise> op/ed/ost/insert/opening/ending" patterns.
    for m in re.finditer(
        r"\b([a-z0-9][a-z0-9\s:+!-]{2,40})\s+"
        r"(op|ed|ost|bgm|opening|ending|insert)\b",
        text,
    ):
        fr = _normalize_text(m.group(1))
        fr = re.sub(r"\b(anime|season|movie|the)\b", " ", fr)
        fr = re.sub(r"\s+", " ", fr).strip()
        if len(fr) >= 4:
            tags.add(f"fr:{fr}")

    # Generic phrase tokens (2-3 grams) from title to catch recurring themes.
    words = [w for w in text.split() if len(w) >= 3 and w not in THEME_STOPWORDS]
    for i in range(len(words)):
        tags.add(f"tk:{words[i]}")
        if i + 1 < len(words):
            tags.add(f"tk:{words[i]} {words[i + 1]}")
        if i + 2 < len(words):
            tags.add(f"tk:{words[i]} {words[i + 1]} {words[i + 2]}")

    return tags


def estimate_title_energy(song_title):
    """Estimate rough track energy from title markers."""
    _, title = split_song(song_title)
    text = _normalize_text(title)
    energy = 0.0
    for marker, w in ENERGY_HINTS.items():
        if marker in text:
            energy += w
    return max(-2.0, min(2.0, energy))


def build_song_features(songs_dic):
    """
    Precompute metadata from full database for better scoring:
      - normalized artist
      - extracted theme tags
      - title-based energy hints
      - corpus-level theme idf
    """
    counts = Counter(songs_dic.values())
    unique_songs = list(counts.keys())
    theme_df = Counter()
    song_features = {}

    for song in unique_songs:
        artist, _ = split_song(song)
        artist_norm = normalize_artist(artist)
        themes = extract_theme_tags(song)
        # Keep only medium-length, interpretable tags.
        themes = {t for t in themes if 5 <= len(t) <= 48}
        for t in themes:
            theme_df[t] += 1
        song_features[song] = {
            "artist_norm": artist_norm,
            "themes": themes,
            "title_energy": estimate_title_energy(song),
            "play_count": counts[song],
        }

    total = max(len(unique_songs), 1)
    theme_idf = {}
    for t, df in theme_df.items():
        if df < 5:
            continue
        # Standard idf-like weight. Higher => more specific theme.
        theme_idf[t] = math.log((1 + total) / (1 + df)) + 1.0

    # Prune song theme sets to tags known in corpus idf map.
    for meta in song_features.values():
        meta["themes"] = {t for t in meta["themes"] if t in theme_idf}

    return song_features, theme_idf


def reconstruct_sessions(songs_dic, dj_dic):
    """
    Reconstruct DJ sessions from dj_dic metadata.

    A session is a consecutive run of plays by the same DJ
    where the gap between plays is < SESSION_GAP_MINUTES.

    Returns list of dicts:
      {
        'dj': str,
        'plays': [(datetime, song_title, ts_str), ...],
        'start': datetime,
        'end': datetime,
        'duration_min': float,
      }
    """
    # Build timeline: [(datetime, song, ts_str, dj_name), ...]
    timeline = []
    for ts_str, dj_name in dj_dic.items():
        if dj_name is None:
            continue
        song = songs_dic.get(ts_str, "")
        try:
            dt = datetime.strptime(ts_str, RADIO_DATE_FMT)
        except ValueError:
            continue
        timeline.append((dt, song, ts_str, dj_name))

    timeline.sort(key=lambda x: x[0])

    # Split into sessions
    sessions = []
    if not timeline:
        return sessions

    current_dj = timeline[0][3]
    current_plays = [(timeline[0][0], timeline[0][1], timeline[0][2])]

    for i in range(1, len(timeline)):
        dt, song, ts_str, dj_name = timeline[i]
        prev_dt = current_plays[-1][0]
        gap = (dt - prev_dt).total_seconds() / 60

        if dj_name == current_dj and 0 < gap < SESSION_GAP_MINUTES:
            current_plays.append((dt, song, ts_str))
        else:
            # End current session
            if len(current_plays) >= 2:
                sessions.append(_make_session(current_dj, current_plays))
            current_dj = dj_name
            current_plays = [(dt, song, ts_str)]

    if len(current_plays) >= 2:
        sessions.append(_make_session(current_dj, current_plays))

    return sessions


def _make_session(dj, plays):
    return {
        "dj": dj,
        "plays": plays,
        "start": plays[0][0],
        "end": plays[-1][0],
        "duration_min": (plays[-1][0] - plays[0][0]).total_seconds() / 60,
    }


def compute_song_rarity(songs_dic):
    """
    Compute rarity score for each song based on overall play frequency.
    Returns dict: song_title -> rarity (0=most common, 1=unique).
    """
    counts = Counter(songs_dic.values())
    max_count = max(counts.values())
    # Log-scale rarity: log(max/count) / log(max)
    import math

    rarity = {}
    log_max = math.log(max_count + 1)
    for song, count in counts.items():
        rarity[song] = math.log((max_count + 1) / (count + 1)) / log_max
    return rarity


def score_arc_shape(plays, song_features):
    """
    Score how well the set exhibits ebb-and-flow energy arcs.

    Uses a blended signal:
      1) pacing changes (gap dynamics)
      2) title-derived energy hints (OP/remix/live vs OST/acoustic/etc)
      3) peak placement (avoid strongest energy only at edges)

    This prevents slow cinematic sets from being penalized just for
    low track density.

    Returns score 0-10.
    """
    n = len(plays)
    if n < 20:
        return 5.0  # Not enough data for arc analysis

    # Compute inter-song gaps in minutes
    gaps = []
    for i in range(1, n):
        g = (plays[i][0] - plays[i - 1][0]).total_seconds() / 60
        if g > 0:
            gaps.append(g)
        else:
            gaps.append(0.1)  # small positive for zero gaps

    if len(gaps) < 15:
        return 5.0

    # Sliding window: compute local pacing (songs per window)
    window = max(5, n // 8)
    local_rates = []
    for i in range(len(gaps) - window + 1):
        window_gaps = gaps[i : i + window]
        # Songs per hour in this window
        total_time = sum(window_gaps)
        if total_time > 0:
            rate = window * 60 / total_time
        else:
            rate = window * 60  # very fast
        local_rates.append(rate)

    if len(local_rates) < 4:
        return 5.0

    # Measure pacing variability — we WANT moderate variability.
    mean_rate = statistics.mean(local_rates)
    if mean_rate == 0:
        return 5.0
    cv = statistics.stdev(local_rates) / mean_rate

    # Count direction changes (peaks and valleys)
    direction_changes = 0
    for i in range(2, len(local_rates)):
        prev_dir = local_rates[i - 1] - local_rates[i - 2]
        curr_dir = local_rates[i] - local_rates[i - 1]
        if prev_dir * curr_dir < 0:
            direction_changes += 1

    changes_per_hour = direction_changes / (sum(gaps) / 60) if sum(gaps) > 0 else 0

    # Base pacing arc.
    cv_score = _bell_score(cv, ideal=0.25, sigma=0.15)
    change_score = _bell_score(changes_per_hour, ideal=3.0, sigma=2.0)

    # Title-energy arc signal (independent from timing gaps).
    title_energy = [
        song_features.get(song, {}).get("title_energy", 0.0) for _, song, _ in plays
    ]
    if len(title_energy) >= 5:
        smoothed = []
        smooth_w = 3
        for i in range(len(title_energy)):
            lo = max(0, i - smooth_w)
            hi = min(len(title_energy), i + smooth_w + 1)
            smoothed.append(statistics.mean(title_energy[lo:hi]))
        rng = max(smoothed) - min(smoothed)
        rng_score = _bell_score(rng, ideal=1.2, sigma=0.9)

        t_changes = 0
        for i in range(2, len(smoothed)):
            prev_dir = smoothed[i - 1] - smoothed[i - 2]
            curr_dir = smoothed[i] - smoothed[i - 1]
            if prev_dir * curr_dir < 0:
                t_changes += 1
        t_changes_per_hour = t_changes / (sum(gaps) / 60) if sum(gaps) > 0 else 0
        t_change_score = _bell_score(t_changes_per_hour, ideal=2.0, sigma=1.5)

        peak_idx = max(range(len(smoothed)), key=lambda i: smoothed[i])
        peak_pos = peak_idx / max(len(smoothed) - 1, 1)
        peak_score = _bell_score(peak_pos, ideal=0.65, sigma=0.30)
    else:
        rng_score = 0.5
        t_change_score = 0.5
        peak_score = 0.5

    return (
        cv_score * 3
        + change_score * 2
        + rng_score * 2
        + t_change_score * 2
        + peak_score * 1
    )


def _bell_score(value, ideal, sigma):
    """Gaussian bell curve scoring: 1.0 at ideal, decaying with distance."""
    return math.exp(-0.5 * ((value - ideal) / sigma) ** 2)


def _max_bell_score(value, centers):
    """Best-of bell scores for multi-style targets."""
    return max(_bell_score(value, ideal=c, sigma=s) for c, s in centers)


def _theme_overlap_score(themes_a, themes_b, theme_idf):
    overlap = themes_a & themes_b
    if not overlap:
        return 0.0
    num = sum(theme_idf.get(t, 1.0) for t in overlap)
    den = sum(theme_idf.get(t, 1.0) for t in (themes_a | themes_b))
    if den <= 0:
        return 0.0
    return num / den


def score_block_coherence(plays, song_features, theme_idf):
    """
    Score how well the DJ groups songs into thematic blocks.

    Looks for runs of related songs:
      - same normalized artist
      - shared franchise/theme tags from title extraction

    Returns score 0-10.
    """
    n = len(plays)
    if n < 10:
        return 5.0

    # Count related runs (2+ consecutive songs by artist or shared theme).
    run_lengths = []
    strong_run_lengths = []
    current_run = 1
    current_strength = 0.0

    for i in range(1, n):
        prev_song = plays[i - 1][1]
        song = plays[i][1]
        prev_meta = song_features.get(prev_song, {})
        meta = song_features.get(song, {})
        same_artist = prev_meta.get("artist_norm") and prev_meta.get(
            "artist_norm"
        ) == meta.get("artist_norm")
        theme_score = _theme_overlap_score(
            prev_meta.get("themes", set()),
            meta.get("themes", set()),
            theme_idf,
        )
        rel_strength = 1.0 if same_artist else theme_score

        if rel_strength >= 0.18:
            current_run += 1
            current_strength += rel_strength
        else:
            if current_run >= 2:
                run_lengths.append(current_run)
                avg_strength = current_strength / max(current_run - 1, 1)
                if avg_strength >= 0.45:
                    strong_run_lengths.append(current_run)
            current_run = 1
            current_strength = 0.0

    if current_run >= 2:
        run_lengths.append(current_run)
        avg_strength = current_strength / max(current_run - 1, 1)
        if avg_strength >= 0.45:
            strong_run_lengths.append(current_run)

    # Percentage of songs in related runs.
    songs_in_runs = sum(run_lengths)
    run_ratio = songs_in_runs / n

    # Longer blocks indicate intentional programming.
    long_blocks = [r for r in run_lengths if r >= 3]
    strong_long_blocks = [r for r in strong_run_lengths if r >= 3]

    # Sweet spot: some blocks, but not monolithic.
    run_score = _bell_score(run_ratio, ideal=0.28, sigma=0.18)
    strong_bonus = min(sum(strong_long_blocks) / max(n * 0.25, 1), 1.0)

    block_bonus = min(len(long_blocks) / max(n / 20, 1), 1.0)

    return run_score * 6 + block_bonus * 2 + strong_bonus * 2


def score_diversity(plays, songs_dic):
    """
    Score artist and song diversity within the set.

    Returns score 0-10.
    """
    n = len(plays)
    if n < 10:
        return 5.0

    songs = [song for _, song, _ in plays]
    artists = [extract_artist(song) for song in songs]

    unique_artists = len(set(artists))
    unique_songs = len(set(songs))

    artist_ratio = unique_artists / n
    song_ratio = unique_songs / n

    # Good diversity: 70-95% unique artists
    artist_score = _bell_score(artist_ratio, ideal=0.85, sigma=0.15)
    # Song uniqueness: should be near 100% (no repeats within a set)
    song_score = min(song_ratio, 1.0)

    return artist_score * 6 + song_score * 4


def score_pacing(plays):
    """
    Score pacing consistency and variety.

    Good DJs maintain a deliberate pace — neither rushed nor dragging.
    The ideal songs/hour depends on style (Claud ~15, Suzubrah ~28),
    but consistency within style matters.

    Returns score 0-10.
    """
    n = len(plays)
    if n < 10:
        return 5.0

    gaps = []
    for i in range(1, n):
        g = (plays[i][0] - plays[i - 1][0]).total_seconds() / 60
        if g > 0:
            gaps.append(g)

    if len(gaps) < 5:
        return 5.0

    mean_gap = statistics.mean(gaps)

    # Songs per hour
    total_hours = sum(gaps) / 60
    if total_hours == 0:
        return 5.0
    songs_per_hour = n / total_hours

    # Gap consistency (CV) — lower is more deliberate
    gap_cv = statistics.stdev(gaps) / mean_gap if mean_gap > 0 else 99

    # Ideal CV: 0.2-0.5 (some variation but not chaotic)
    cv_score = _bell_score(gap_cv, ideal=0.35, sigma=0.20)

    # Two healthy style peaks:
    #   - cinematic arc sets (~15/hr)
    #   - dense collage sets (~28/hr)
    rate_score = _max_bell_score(songs_per_hour, centers=[(15, 4.5), (28, 5.5)])

    # Style-aware consistency target.
    slow_fit = _bell_score(songs_per_hour, ideal=15, sigma=5)
    fast_fit = _bell_score(songs_per_hour, ideal=28, sigma=6)
    slow_w = slow_fit / max(slow_fit + fast_fit, 1e-9)
    fast_w = 1.0 - slow_w
    cv_target = slow_w * 0.30 + fast_w * 0.45
    cv_score = _bell_score(gap_cv, ideal=cv_target, sigma=0.20)

    return cv_score * 6 + rate_score * 4


def score_novelty(plays, song_rarity):
    """
    Score how many deep cuts vs overplayed songs the set has.

    Returns score 0-10.
    """
    n = len(plays)
    if n < 10:
        return 5.0

    rarities = []
    for _, song, _ in plays:
        base_rarity = song_rarity.get(song, 0.5)
        # Bootleg/remix versions of common songs should be treated as more novel —
        # the DJ curated a *version*, not the common original.
        lower = song.lower()
        if base_rarity < 0.3 and any(kw in lower for kw in REMIX_KEYWORDS):
            base_rarity = max(
                base_rarity, 0.5
            )  # floor at 0.5 for remixes of popular songs
        rarities.append(base_rarity)

    avg_rarity = statistics.mean(rarities)

    # Novelty should be monotonically rewarding — playing rarer music should
    # NEVER be penalized. A DJ who plays almost all underground tracks (like
    # sauce) should score highest, not get penalized for "too rare".
    #
    # Scale: 0.0 avg_rarity = 0 score, 1.0 avg_rarity = 10.0 score
    # Use a slight curve that rewards moderate rarity well but keeps scaling up.
    rarity_score = min(avg_rarity * 1.2, 1.0)  # saturates at ~0.83

    # Bonus for deep cut density (tracks played 1-3 times ever)
    deep_ratio = sum(1 for r in rarities if r > 0.78) / n
    deep_bonus = min(deep_ratio * 1.5, 1.0)  # 67%+ deep cuts = max bonus

    # Keep novelty focused on rarity/deep-cuts only.
    # Mainstream usage gets its own explicit penalty dimension.
    raw = rarity_score * 8 + deep_bonus * 2
    return max(0.0, min(raw, 10.0))


def _songs_per_hour(plays):
    """Compute songs/hour from observed inter-song timing."""
    n = len(plays)
    if n < 2:
        return 0.0
    gaps = []
    for i in range(1, n):
        g = (plays[i][0] - plays[i - 1][0]).total_seconds() / 60
        if g > 0:
            gaps.append(g)
    if not gaps:
        return 0.0
    total_hours = sum(gaps) / 60
    if total_hours <= 0:
        return 0.0
    return n / total_hours


def _mainstream_pct(plays, song_rarity, threshold=0.22):
    """
    Fraction of tracks that are highly popular/common in the corpus.

    Tracks with rarity below threshold are treated as mainstream picks.
    """
    if not plays:
        return 0.0
    mainstream = 0
    for _, song, _ in plays:
        if song_rarity.get(song, 0.5) < threshold:
            mainstream += 1
    return mainstream / len(plays)


def score_pace_density(plays):
    """
    Score set density from songs/hour using a bell curve centered at 25.

    Returns score 0-10.
    """
    n = len(plays)
    if n < 10:
        return 5.0
    sph = _songs_per_hour(plays)
    return _bell_score(sph, ideal=25.0, sigma=8.0) * 10.0


def score_mainstream_penalty(plays, song_rarity):
    """
    Penalty score (0-10) for high mainstream share.

    No penalty <=20% mainstream, then ramps quickly after 35%.
    """
    n = len(plays)
    if n < 10:
        return 0.0

    mainstream_pct = _mainstream_pct(plays, song_rarity, threshold=0.22)
    if mainstream_pct <= 0.20:
        return 0.0
    if mainstream_pct >= 0.80:
        return 10.0

    x = (mainstream_pct - 0.20) / 0.60  # 0..1
    return min((x**1.35) * 10.0, 10.0)


def score_remix_curation(plays):
    """
    Score how much the DJ curates through bootlegs/remixes and scene knowledge.

    This rewards DJs who don't just play original tracks but seek out
    (or create) remixed/bootleg versions that are club-ready. This is
    a sign of actual DJ skill and deep underground scene knowledge.

    Also rewards presence of known doujin/electronic scene artists.

    Returns score 0-10.
    """
    n = len(plays)
    if n < 10:
        return 5.0

    remix_count = 0
    scene_count = 0

    for _, song, _ in plays:
        lower = song.lower()
        # Check for remix/bootleg keywords
        if any(kw in lower for kw in REMIX_KEYWORDS):
            remix_count += 1
        # Check for scene artists
        artist = extract_artist(song).lower()
        if any(sa in artist for sa in SCENE_ARTISTS_LOWER):
            scene_count += 1

    remix_ratio = remix_count / n
    scene_ratio = scene_count / n

    # Reward high remix density — sauce's gold standard is ~48%, most DJs are 6-16%.
    # Use a plateau: anything above 20% is increasingly good, maxes around 40-50%.
    if remix_ratio >= 0.20:
        remix_score = min(remix_ratio / 0.45, 1.0)
    else:
        remix_score = remix_ratio / 0.20 * 0.5  # some credit for lower amounts

    # Scene knowledge bonus — having 10%+ underground artists is increasingly good
    if scene_ratio >= 0.08:
        scene_score = min(scene_ratio / 0.25, 1.0)
    else:
        scene_score = scene_ratio / 0.08 * 0.3

    # Combined: 60% remix, 40% scene
    return remix_score * 6 + scene_score * 4


def score_production_coherence(plays):
    """
    Score how coherent the production quality across the set is.

    Audio analysis proved: 100% bootleg/remix sets achieve 0.970 timbre
    continuity vs 0.875 for mixed original/remix sets. This is because
    modern bootleg production uses consistent mastering chains, sidechaining,
    and digital production that original 90s/00s anime OSTs lack.

    This dimension rewards DJs who curate through production style,
    not just track selection.

    Returns score 0-10.
    """
    n = len(plays)
    if n < 10:
        return 5.0

    remix_count = 0
    scene_count = 0
    for _, song, _ in plays:
        lower = song.lower()
        if any(kw in lower for kw in REMIX_KEYWORDS):
            remix_count += 1
        artist = extract_artist(song).lower()
        if any(sa in artist for sa in SCENE_ARTISTS_LOWER):
            scene_count += 1

    remix_ratio = remix_count / n
    scene_ratio = scene_count / n
    production_ratio = min(remix_ratio + scene_ratio, 1.0)

    # Audio-validated insight: high production_ratio correlates with
    # timbral consistency (r~0.9). sauce's 100% bootleg set = 0.970 timbre.
    # Claud's mixed set = 0.875 timbre.
    #
    # Score: 0% modern production = 3.0, 100% = 10.0
    # This is NOT about bootlegs being "better" — it's about production
    # coherence creating smoother transitions.
    if production_ratio >= 0.60:
        return 8.0 + min((production_ratio - 0.60) / 0.40, 1.0) * 2.0
    elif production_ratio >= 0.30:
        return 5.0 + (production_ratio - 0.30) / 0.30 * 3.0
    else:
        return 3.0 + production_ratio / 0.30 * 2.0


def score_length(duration_min):
    """
    Bonus score for set length. Sweet spot: 2-6 hours.

    Returns score 0-5.
    """
    hours = duration_min / 60
    if hours < 0.5:
        return 0.0
    elif hours < 1:
        return 1.0
    elif hours < 2:
        return 2.0 + (hours - 1) * 1.0
    elif hours <= 6:
        return 3.0 + min((hours - 2) / 4, 1.0) * 2.0
    else:
        # Diminishing returns past 6 hours
        return 5.0


def score_set(session, song_rarity, songs_dic, song_features, theme_idf):
    """
    Compute per-dimension descriptors for a DJ session.

    Each dimension is scored independently (0-10 scale). No overall
    score — dimensions are descriptors, not components of a total.
    """
    plays = session["plays"]

    arc = score_arc_shape(plays, song_features)
    blocks = score_block_coherence(plays, song_features, theme_idf)
    diversity = score_diversity(plays, songs_dic)
    pacing = score_pacing(plays)
    novelty = score_novelty(plays, song_rarity)
    pace_density = score_pace_density(plays)
    mainstream_penalty = score_mainstream_penalty(plays, song_rarity)
    remix = score_remix_curation(plays)
    production = score_production_coherence(plays)
    length = score_length(session["duration_min"])

    songs_per_hour = _songs_per_hour(plays)
    mainstream_pct = _mainstream_pct(plays, song_rarity, threshold=0.22)

    return {
        "arc_shape": round(arc, 1),
        "block_coherence": round(blocks, 1),
        "diversity": round(diversity, 1),
        "pacing": round(pacing, 1),
        "novelty": round(novelty, 1),
        "pace_density": round(pace_density, 1),
        "mainstream_penalty": round(mainstream_penalty, 1),
        "songs_per_hour": round(songs_per_hour, 1),
        "mainstream_pct": round(mainstream_pct * 100, 1),
        "remix_curation": round(remix, 1),
        "production_coherence": round(production, 1),
        "length_bonus": round(length, 1),
    }


SORTABLE_DIMENSIONS = [
    "novelty",
    "pace_density",
    "remix_curation",
    "production_coherence",
    "arc_shape",
    "block_coherence",
    "diversity",
    "pacing",
    "mainstream_penalty",
    "length_bonus",
]


def print_session_detail(session, scores, rank, verbose=False):
    """Print a scored session with tracklist."""
    plays = session["plays"]
    n = len(plays)
    artists = [extract_artist(song) for _, song, _ in plays]
    unique_artists = len(set(artists))
    hours = session["duration_min"] / 60
    songs_per_hour = n / hours if hours > 0 else 0

    print(f"\n{'=' * 90}")
    print(
        f"  #{rank}  {session['dj']:20s}  "
        f"{session['start'].strftime('%Y-%m-%d %H:%M')}  "
        f"{n} songs  {hours:.1f}h  {songs_per_hour:.0f} songs/hr"
    )
    print(
        f"  novelty={scores['novelty']}  pace={scores['pace_density']}  "
        f"remix={scores['remix_curation']}  prod={scores.get('production_coherence', '-')}  "
        f"arc={scores['arc_shape']}  blocks={scores['block_coherence']}  "
        f"diversity={scores['diversity']}  pacing={scores['pacing']}  "
        f"length={scores['length_bonus']}  mainstream={scores['mainstream_pct']}%"
    )
    print(f"  {unique_artists} unique artists")
    print(f"{'=' * 90}")

    if verbose:
        for i, (dt, song, _) in enumerate(plays):
            # Mark artist runs
            marker = ""
            if i > 0 and artists[i] == artists[i - 1]:
                marker = " <<"
            print(f"  {i + 1:>3}. {dt.strftime('%H:%M')} {song[:80]}{marker}")
    else:
        # Show first 10, last 5
        show_start = min(10, n)
        show_end = min(5, n)
        for i in range(show_start):
            dt, song, _ = plays[i]
            print(f"  {i + 1:>3}. {dt.strftime('%H:%M')} {song[:80]}")
        if n > show_start + show_end:
            print(f"  ... ({n - show_start - show_end} more songs)")
        if n > show_start:
            for i in range(max(show_start, n - show_end), n):
                dt, song, _ = plays[i]
                print(f"  {i + 1:>3}. {dt.strftime('%H:%M')} {song[:80]}")


def cmd_overview(sessions, songs_dic, song_rarity):
    """Print an overview of all DJs and their sessions."""
    print("=" * 80)
    print("R/A/DIO DJ OVERVIEW")
    print("=" * 80)

    # Filter to human DJs (exclude Hanyuu-sama bot)
    human_sessions = [s for s in sessions if s["dj"] != "Hanyuu-sama"]
    bot_sessions = [s for s in sessions if s["dj"] == "Hanyuu-sama"]

    print(f"\nTotal sessions reconstructed: {len(sessions):,}")
    print(f"  Human DJ sessions: {len(human_sessions):,}")
    print(f"  Hanyuu-sama (bot): {len(bot_sessions):,}")

    # Per-DJ stats
    dj_stats = defaultdict(lambda: {"sets": 0, "songs": 0, "hours": 0.0})
    for s in human_sessions:
        dj = s["dj"]
        dj_stats[dj]["sets"] += 1
        dj_stats[dj]["songs"] += len(s["plays"])
        dj_stats[dj]["hours"] += s["duration_min"] / 60

    # Sort by total hours
    sorted_djs = sorted(dj_stats.items(), key=lambda x: x[1]["hours"], reverse=True)

    print(
        f"\n{'DJ':25s} {'Sets':>6} {'Songs':>8} {'Hours':>8} {'Avg Songs/Set':>14} {'Avg Hours/Set':>14}"
    )
    print("-" * 85)
    for dj, stats in sorted_djs:
        avg_songs = stats["songs"] / stats["sets"] if stats["sets"] > 0 else 0
        avg_hours = stats["hours"] / stats["sets"] if stats["sets"] > 0 else 0
        print(
            f"  {dj:23s} {stats['sets']:>6} {stats['songs']:>8,} {stats['hours']:>8.0f} "
            f"{avg_songs:>14.0f} {avg_hours:>14.1f}"
        )

    # Scoreable sets
    scoreable = [s for s in human_sessions if len(s["plays"]) >= MIN_SET_SONGS]
    print(f"\nSets with {MIN_SET_SONGS}+ songs (scoreable): {len(scoreable):,}")

    return scoreable


def cmd_sort(
    sessions,
    songs_dic,
    song_rarity,
    song_features,
    theme_idf,
    n=30,
    dj_filter=None,
    sort_by="date",
    verbose=False,
):
    """List scored DJ sets, sorted by date or a chosen dimension."""
    candidates = [
        s
        for s in sessions
        if s["dj"] != "Hanyuu-sama" and len(s["plays"]) >= MIN_SET_SONGS
    ]

    if dj_filter:
        dj_lower = dj_filter.lower()
        candidates = [s for s in candidates if s["dj"].lower() == dj_lower]
        if not candidates:
            print(f"No sets found for DJ '{dj_filter}'")
            return

    print(f"\nScoring {len(candidates):,} sets...")

    scored = []
    for session in candidates:
        scores = score_set(session, song_rarity, songs_dic, song_features, theme_idf)
        scored.append((session, scores))

    if sort_by == "date":
        scored.sort(key=lambda x: x[0]["start"], reverse=True)
    else:
        # For mainstream_penalty, lower = better (less mainstream), so don't reverse
        reverse = sort_by != "mainstream_penalty"
        scored.sort(key=lambda x: x[1].get(sort_by, 0), reverse=reverse)

    title = f"DJ SETS — sorted by {sort_by}"
    if dj_filter:
        title += f" — {dj_filter}"
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")

    for rank, (session, scores) in enumerate(scored[:n], 1):
        print_session_detail(session, scores, rank, verbose=verbose)

    return scored


def cmd_dj_profile(sessions, songs_dic, song_rarity, song_features, theme_idf, dj_name):
    """Deep profile of a single DJ's style across all their sets."""
    dj_sessions = [
        s
        for s in sessions
        if s["dj"].lower() == dj_name.lower() and len(s["plays"]) >= MIN_SET_SONGS
    ]

    if not dj_sessions:
        print(f"No scoreable sets found for '{dj_name}'")
        return

    # Score all sets
    scored = []
    for session in dj_sessions:
        scores = score_set(session, song_rarity, songs_dic, song_features, theme_idf)
        scored.append((session, scores))

    scored.sort(key=lambda x: x[0]["start"], reverse=True)

    # Profile stats
    all_songs = []
    all_artists = []
    all_durations = []
    all_paces = []
    for session in dj_sessions:
        plays = session["plays"]
        all_durations.append(session["duration_min"])
        for _, song, _ in plays:
            all_songs.append(song)
            all_artists.append(extract_artist(song))
        if session["duration_min"] > 0:
            all_paces.append(len(plays) / (session["duration_min"] / 60))

    artist_counts = Counter(all_artists)

    print(f"\n{'=' * 80}")
    print(f"DJ PROFILE: {dj_name}")
    print(f"{'=' * 80}")

    print(f"\n  Total sets: {len(dj_sessions)}")
    print(f"  Total songs played: {len(all_songs):,}")
    print(f"  Unique songs: {len(set(all_songs)):,}")
    print(f"  Unique artists: {len(set(all_artists)):,}")
    print(f"  Total hours on air: {sum(all_durations) / 60:.0f}")
    print(
        f"  Avg set length: {statistics.mean(all_durations) / 60:.1f}h "
        f"(median {statistics.median(all_durations) / 60:.1f}h)"
    )
    print(
        f"  Avg pace: {statistics.mean(all_paces):.0f} songs/hr "
        f"(median {statistics.median(all_paces):.0f})"
    )

    # Score averages
    score_keys = [
        "novelty",
        "pace_density",
        "remix_curation",
        "production_coherence",
        "arc_shape",
        "block_coherence",
        "diversity",
        "pacing",
    ]
    print(f"\n  Score averages across {len(scored)} sets:")
    for key in score_keys:
        vals = [s[key] for _, s in scored]
        print(
            f"    {key:20s}: {statistics.mean(vals):5.1f} "
            f"(min {min(vals):.1f}, max {max(vals):.1f})"
        )

    # Top artists
    print("\n  Top 20 most-played artists:")
    for artist, count in artist_counts.most_common(20):
        pct = count / len(all_songs) * 100
        print(f"    [{count:>4}x {pct:4.1f}%] {artist[:60]}")

    # Signature songs (played in >30% of sets)
    n_sets = len(dj_sessions)
    song_set_counts = defaultdict(int)
    for session in dj_sessions:
        seen = set()
        for _, song, _ in session["plays"]:
            if song not in seen:
                song_set_counts[song] += 1
                seen.add(song)

    signatures = [
        (song, count) for song, count in song_set_counts.items() if count / n_sets > 0.3
    ]
    signatures.sort(key=lambda x: x[1], reverse=True)

    if signatures:
        print("\n  Signature songs (in >30% of sets):")
        for song, count in signatures[:20]:
            pct = count / n_sets * 100
            print(f"    [{count:>3}/{n_sets} sets = {pct:.0f}%] {song[:70]}")

    # Recent and earliest sets
    print("\n  5 most recent sets:")
    for rank, (session, scores) in enumerate(scored[:5], 1):
        print_session_detail(session, scores, rank, verbose=False)

    if len(scored) > 5:
        print("\n  3 earliest sets:")
        for rank, (session, scores) in enumerate(scored[-3:], len(scored) - 2):
            print_session_detail(session, scores, rank, verbose=False)


def main():
    parser = argparse.ArgumentParser(description="r/a/dio DJ set analysis")
    parser.add_argument(
        "command",
        choices=["overview", "sort", "profile", "export"],
        help="overview: DJ stats | sort: list sets | profile: DJ deep-dive | export: JSON export",
    )
    parser.add_argument("--dj", type=str, default=None, help="Filter by DJ name")
    parser.add_argument(
        "-n", type=int, default=30, help="Number of results (default 30)"
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="date",
        choices=["date"] + SORTABLE_DIMENSIONS,
        help="Dimension to sort by (default: date)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show full tracklists"
    )
    args = parser.parse_args()

    print("Loading database...")
    songs_dic, dj_dic = load_db()
    print(f"  {len(songs_dic):,} songs, {len(dj_dic):,} DJ entries")

    print("Computing song rarity...")
    song_rarity = compute_song_rarity(songs_dic)
    print("Extracting song themes/features...")
    song_features, theme_idf = build_song_features(songs_dic)

    print("Reconstructing DJ sessions...")
    sessions = reconstruct_sessions(songs_dic, dj_dic)
    print(f"  {len(sessions):,} sessions reconstructed")

    human = [s for s in sessions if s["dj"] != "Hanyuu-sama"]
    print(f"  {len(human):,} human DJ sessions")

    if args.command == "overview":
        cmd_overview(sessions, songs_dic, song_rarity)
    elif args.command == "sort":
        cmd_sort(
            sessions,
            songs_dic,
            song_rarity,
            song_features,
            theme_idf,
            n=args.n,
            dj_filter=args.dj,
            sort_by=args.sort,
            verbose=args.verbose,
        )
    elif args.command == "profile":
        if not args.dj:
            print("Error: --dj required for profile command")
            sys.exit(1)
        cmd_dj_profile(
            sessions, songs_dic, song_rarity, song_features, theme_idf, args.dj
        )
    elif args.command == "export":
        # Export scored sets to JSON for notebook consumption
        candidates = [
            s
            for s in sessions
            if s["dj"] != "Hanyuu-sama" and len(s["plays"]) >= MIN_SET_SONGS
        ]
        if args.dj:
            dj_lower = args.dj.lower()
            candidates = [s for s in candidates if s["dj"].lower() == dj_lower]

        scored = []
        for session in candidates:
            scores = score_set(
                session, song_rarity, songs_dic, song_features, theme_idf
            )
            scored.append(
                {
                    "dj": session["dj"],
                    "start": session["start"].strftime(RADIO_DATE_FMT),
                    "end": session["end"].strftime(RADIO_DATE_FMT),
                    "n_songs": len(session["plays"]),
                    "duration_min": round(session["duration_min"], 1),
                    "scores": scores,
                    "tracklist": [
                        {"time": dt.strftime("%H:%M"), "song": song}
                        for dt, song, _ in session["plays"]
                    ],
                }
            )

        scored.sort(key=lambda x: x["start"])
        out_path = os.path.join(PWD, "dj_sets.json")
        with open(out_path, "w") as f:
            json.dump(scored, f, ensure_ascii=False, indent=2)
        print(f"Exported {len(scored)} scored sets to {out_path}")


if __name__ == "__main__":
    main()
