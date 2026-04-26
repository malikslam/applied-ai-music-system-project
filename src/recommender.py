"""
Core recommender logic for the Applied AI Music System.

Public API:
  Song, UserProfile         — data classes
  Recommender               — OOP interface with score_all(), recommend(), explain_recommendation()
  load_songs()              — CSV loader returning List[Dict]
  score_song()              — weighted similarity scorer
  recommend_songs()         — functional top-k wrapper (kept for backward compat)
"""
import csv
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .knowledge_base import retrieve_context, _REASON_FIRST_WORD_TO_FEATURE

# ---------------------------------------------------------------------------
# Feature weights  (see model_card.md for rationale)
# Mood is 2x genre: one genre spans many moods, but mood directly signals feel.
# ---------------------------------------------------------------------------
WEIGHTS: Dict[str, float] = {
    "mood":         2.0,
    "energy":       1.5,
    "acousticness": 1.5,
    "genre":        1.0,
    "tempo_bpm":    1.0,
    "valence":      1.0,
    "danceability": 0.5,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _profile_to_dict(user: UserProfile) -> Dict:
    """Convert a UserProfile to the dict format expected by score_song()."""
    return {
        "genre":        user.favorite_genre,
        "mood":         user.favorite_mood,
        "energy":       user.target_energy,
        "acousticness": 0.85 if user.likes_acoustic else 0.15,
    }


def _song_to_dict(song: Song) -> Dict:
    """Convert a Song dataclass to the dict format expected by score_song()."""
    return {
        "genre":        song.genre,
        "mood":         song.mood,
        "energy":       song.energy,
        "acousticness": song.acousticness,
        "valence":      song.valence,
        "tempo_bpm":    song.tempo_bpm,
        "danceability": song.danceability,
    }


def _features_from_reasons(reasons: List[str]) -> List[str]:
    """
    Extract FEATURE_DOCS keys from score_song() reason strings.

    Reason strings start with the feature name (e.g. "energy is a close match").
    Maps "tempo" → "tempo_bpm" via _REASON_FIRST_WORD_TO_FEATURE.
    """
    found: List[str] = []
    for reason in reasons:
        first = reason.split()[0].lower()
        key = _REASON_FIRST_WORD_TO_FEATURE.get(first)
        if key and key not in found:
            found.append(key)
    return found


def _fallback_explanation(song: Song, score: float, reasons: List[str]) -> str:
    reason_str = ", ".join(reasons) if reasons else "partial feature similarity"
    return f"{song.title} by {song.artist} scored {score:.2f}. Match signals: {reason_str}."


def _generate_explanation(
    user: UserProfile,
    song: Song,
    score: float,
    reasons: List[str],
    context: str,
) -> str:
    """
    Call Claude (Haiku) with the score breakdown + retrieved context to generate
    a natural-language explanation. Raises EnvironmentError if no API key is set.
    """
    import anthropic  # local import so missing package doesn't break other imports

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set — falling back to template.")

    client = anthropic.Anthropic(api_key=api_key)

    reasons_text = "\n".join(f"- {r}" for r in reasons) if reasons else "- partial feature similarity"

    prompt = f"""You are a music recommendation assistant explaining why a song was recommended.

USER PROFILE:
- Preferred genre: {user.favorite_genre}
- Preferred mood: {user.favorite_mood}
- Target energy: {user.target_energy}
- Prefers acoustic: {user.likes_acoustic}

RECOMMENDED SONG:
- Title: "{song.title}" by {song.artist}
- Genre: {song.genre} | Mood: {song.mood}
- Energy: {song.energy} | Acousticness: {song.acousticness}
- Valence: {song.valence} | Tempo: {song.tempo_bpm} BPM
- Match score: {score:.2f} / 1.00

SCORE BREAKDOWN (from scoring algorithm):
{reasons_text}

RETRIEVED MUSIC KNOWLEDGE:
{context}

Write 2–3 sentences explaining why this song was recommended. Reference specific matched \
features from the score breakdown. Be honest if it is only a partial match. \
Do not start your response with "I"."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=180,
        messages=[{"role": "user", "content": prompt}],
    )

    explanation = message.content[0].text.strip()

    # Validation: if score is weak but LLM overclaims, append a correction note
    overclaim_words = ("perfect match", "excellent match", "ideal match")
    if score < 0.4 and any(w in explanation.lower() for w in overclaim_words):
        explanation += f" (Note: overall match score is {score:.2f} — this is a partial match.)"

    return explanation


# ---------------------------------------------------------------------------
# Recommender class
# ---------------------------------------------------------------------------

class Recommender:
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def score_all(self, user: UserProfile) -> List[Tuple[Song, float]]:
        """Score every song against the user profile, sorted highest-first."""
        user_prefs = _profile_to_dict(user)
        scored = [
            (song, score_song(user_prefs, _song_to_dict(song))[0])
            for song in self.songs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs for the given user profile."""
        return [song for song, _ in self.score_all(user)[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """
        Return a RAG-enriched explanation of why song was recommended to user.

        Flow: score_song() → retrieve_context() → Claude API → validate.
        Falls back to a template string if the API key is missing or the call fails.
        """
        user_prefs = _profile_to_dict(user)
        score, reasons = score_song(user_prefs, _song_to_dict(song))

        top_features = _features_from_reasons(reasons)
        context = retrieve_context(song.genre, song.mood, top_features)

        try:
            return _generate_explanation(user, song, score, reasons, context)
        except Exception:
            return _fallback_explanation(song, score, reasons)


# ---------------------------------------------------------------------------
# Functional interface (kept for backward compatibility)
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """Parse songs.csv and return a list of song dicts with typed fields."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Return (score 0–1, reasons) by comparing song features to user_prefs."""
    feature_scores: Dict[str, float] = {}
    reasons: List[str] = []

    if "genre" in user_prefs:
        match = 1.0 if user_prefs["genre"] == song["genre"] else 0.0
        feature_scores["genre"] = match
        if match:
            reasons.append(f"genre matched ({song['genre']})")

    if "mood" in user_prefs:
        match = 1.0 if user_prefs["mood"] == song["mood"] else 0.0
        feature_scores["mood"] = match
        if match:
            reasons.append(f"mood matched ({song['mood']})")

    if "energy" in user_prefs:
        s = 1.0 - abs(user_prefs["energy"] - song["energy"])
        feature_scores["energy"] = s
        if s >= 0.85:
            reasons.append(f"energy is a close match ({song['energy']})")

    if "acousticness" in user_prefs:
        s = 1.0 - abs(user_prefs["acousticness"] - song["acousticness"])
        feature_scores["acousticness"] = s
        if s >= 0.85:
            reasons.append(f"acousticness is a close match ({song['acousticness']})")

    if "valence" in user_prefs:
        s = 1.0 - abs(user_prefs["valence"] - song["valence"])
        feature_scores["valence"] = s

    if "tempo_bpm" in user_prefs:
        user_tempo = user_prefs["tempo_bpm"] / 200.0
        song_tempo = song["tempo_bpm"] / 200.0
        s = 1.0 - abs(user_tempo - song_tempo)
        feature_scores["tempo_bpm"] = s
        if s >= 0.85:
            reasons.append(f"tempo is a close match ({song['tempo_bpm']} BPM)")

    if "danceability" in user_prefs:
        s = 1.0 - abs(user_prefs["danceability"] - song["danceability"])
        feature_scores["danceability"] = s

    total_weighted = sum(feature_scores[f] * WEIGHTS[f] for f in feature_scores)
    total_weights = sum(WEIGHTS[f] for f in feature_scores)
    final_score = total_weighted / total_weights if total_weights > 0 else 0.0

    if not reasons:
        reasons.append("partial numeric similarity, no exact categorical match")

    return final_score, reasons


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5
) -> List[Tuple[Dict, float, str]]:
    """Score every song, sort descending, return top-k (song_dict, score, explanation)."""
    scored = [
        (song, *score_song(user_prefs, song))
        for song in songs
    ]
    # score_song returns (score, reasons); unpack to (song, score, reasons_list)
    result = [
        (song, score, ", ".join(reasons))
        for song, score, reasons in scored
    ]
    result.sort(key=lambda x: x[1], reverse=True)
    return result[:k]
