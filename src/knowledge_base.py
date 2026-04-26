"""
Local RAG knowledge base for the music recommender.

retrieve_context() fetches relevant genre, mood, and feature descriptions
that are injected into the LLM prompt inside explain_recommendation().
"""
from typing import List

GENRE_DOCS = {
    "rock": (
        "Rock music features electric guitars, strong rhythms, and high energy. "
        "Typical characteristics: energy 0.7–1.0, acousticness 0.0–0.3, tempo 120–160 BPM. "
        "Rock prioritizes intensity and emotional power over polish."
    ),
    "pop": (
        "Pop music is designed for broad appeal with catchy melodies and polished production. "
        "Typical characteristics: danceability 0.7–0.9, valence 0.7–0.9, tempo 100–130 BPM. "
        "Pop is the most commercially accessible genre."
    ),
    "lofi": (
        "Lo-fi music creates warm, nostalgic atmospheres ideal for studying or relaxing. "
        "Typical characteristics: energy 0.3–0.5, acousticness 0.7–0.9, tempo 60–90 BPM. "
        "Intentional imperfections like vinyl crackle and tape hiss are part of the aesthetic."
    ),
    "ambient": (
        "Ambient music prioritizes atmosphere and texture over rhythm or melody. "
        "Typical characteristics: energy 0.2–0.4, acousticness 0.7–1.0, tempo 50–70 BPM. "
        "Ambient creates immersive sonic environments with minimal beats."
    ),
    "jazz": (
        "Jazz emphasizes improvisation, complex chord progressions, and swing rhythms. "
        "Typical characteristics: energy 0.3–0.6, acousticness 0.7–0.9, tempo 80–120 BPM. "
        "Jazz rewards attentive listening and musical complexity."
    ),
    "synthwave": (
        "Synthwave evokes 1980s electronic aesthetics with synthesizers and drum machines. "
        "Typical characteristics: energy 0.6–0.8, acousticness 0.1–0.3, tempo 90–120 BPM. "
        "Synthwave is nostalgic yet fully electronic."
    ),
    "indie pop": (
        "Indie pop combines pop accessibility with independent artistic sensibilities. "
        "Typical characteristics: acousticness 0.3–0.6, valence 0.7–0.9, tempo 100–130 BPM. "
        "Indie pop bridges organic warmth and polished catchiness."
    ),
}

MOOD_DOCS = {
    "happy": (
        "Happy music creates joy and optimism. "
        "Typical signals: major keys, upbeat tempo, high valence (0.75+). "
        "Happy tracks encourage movement and positive emotional states."
    ),
    "chill": (
        "Chill music promotes relaxation and a low-stress environment. "
        "Typical signals: gentle rhythms, soft timbres, low energy (0.3–0.5). "
        "Ideal for studying, coffee shops, or winding down."
    ),
    "intense": (
        "Intense music amplifies adrenaline and focus. "
        "Typical signals: fast tempo (130+ BPM), high energy (0.8+), driving rhythms. "
        "Ideal for workouts or moments requiring peak concentration."
    ),
    "moody": (
        "Moody music creates an introspective or emotionally complex atmosphere. "
        "Typical signals: minor keys, mid-range tempo, lower valence (0.3–0.6). "
        "Suited for night drives or creative work."
    ),
    "relaxed": (
        "Relaxed music is unhurried but not sleepy. "
        "Typical signals: acoustic instruments, moderate tempo, warm textures. "
        "Works well for casual social settings or gentle background ambiance."
    ),
    "focused": (
        "Focused music supports cognitive tasks with steady, non-distracting rhythms. "
        "Typical signals: moderate energy (0.35–0.55), consistent tempo. "
        "Creates a stable sonic environment for work or study."
    ),
}

FEATURE_DOCS = {
    "energy": (
        "Energy (0–1) measures perceptual intensity and activity. "
        "High energy (0.8–1.0) = fast, loud, noisy. Low energy (0.0–0.3) = slow, quiet, calm. "
        "Strong predictor of active vs. passive listening context."
    ),
    "acousticness": (
        "Acousticness (0–1) measures natural vs. electronic sound. "
        "High (0.7–1.0) = organic acoustic instruments like guitar or piano. "
        "Low (0.0–0.3) = heavy electronic production or synthesizers."
    ),
    "valence": (
        "Valence (0–1) measures musical positivity. "
        "High (0.7–1.0) = happy, cheerful. Low (0.0–0.3) = sad or dark. "
        "Distinguishes 'happy intense' from 'dark intense' tracks."
    ),
    "danceability": (
        "Danceability (0–1) measures suitability for dancing based on tempo regularity and beat strength. "
        "High (0.7–0.9) = steady, predictable groove. Low (0.0–0.4) = irregular or free-form."
    ),
    "tempo_bpm": (
        "Tempo (BPM) is track speed. Slow (60–80 BPM) = relaxed. "
        "Moderate (90–120 BPM) = natural. Fast (130–160+ BPM) = urgent or intense. "
        "Normalized to 0–1 by dividing by 200 for scoring."
    ),
}

# Maps the first word of a score_song() reason string to a FEATURE_DOCS key
_REASON_FIRST_WORD_TO_FEATURE = {
    "energy": "energy",
    "acousticness": "acousticness",
    "tempo": "tempo_bpm",
    "valence": "valence",
    "danceability": "danceability",
}


def retrieve_context(genre: str, mood: str, top_features: List[str]) -> str:
    """
    Return relevant knowledge base text for the given genre, mood, and feature names.

    top_features should use FEATURE_DOCS keys (e.g. 'energy', 'tempo_bpm').
    Up to 2 feature docs are appended after the genre and mood entries.
    """
    chunks = []

    if genre in GENRE_DOCS:
        chunks.append(f"[Genre: {genre}]\n{GENRE_DOCS[genre]}")

    if mood in MOOD_DOCS:
        chunks.append(f"[Mood: {mood}]\n{MOOD_DOCS[mood]}")

    for feat in top_features[:2]:
        if feat in FEATURE_DOCS:
            chunks.append(f"[Feature: {feat}]\n{FEATURE_DOCS[feat]}")

    return "\n\n".join(chunks)
