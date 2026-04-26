"""Tests for Recommender — recommend() and explain_recommendation()."""
from src.recommender import Song, UserProfile, Recommender


def _small_recommender() -> Recommender:
    return Recommender([
        Song(id=1, title="Test Pop Track", artist="Test Artist",
             genre="pop", mood="happy", energy=0.8, tempo_bpm=120,
             valence=0.9, danceability=0.8, acousticness=0.2),
        Song(id=2, title="Chill Lofi Loop", artist="Test Artist",
             genre="lofi", mood="chill", energy=0.4, tempo_bpm=80,
             valence=0.6, danceability=0.5, acousticness=0.9),
    ])


def test_recommend_returns_correct_count():
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)
    rec = _small_recommender()
    assert len(rec.recommend(user, k=2)) == 2


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)
    rec = _small_recommender()
    results = rec.recommend(user, k=2)
    # pop/happy should score higher than lofi/chill for this profile
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_recommend_k_less_than_catalog():
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)
    rec = _small_recommender()
    assert len(rec.recommend(user, k=1)) == 1


def test_score_all_returns_all_songs_sorted():
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)
    rec = _small_recommender()
    scored = rec.score_all(user)
    assert len(scored) == 2
    # scores should be descending
    assert scored[0][1] >= scored[1][1]


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)
    rec = _small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_explain_recommendation_fallback_contains_title():
    """Fallback explanation must reference the song title when API is unavailable."""
    import unittest.mock as mock
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)
    rec = _small_recommender()
    song = rec.songs[0]
    # Force _generate_explanation to raise so fallback is used
    with mock.patch("src.recommender._generate_explanation", side_effect=Exception("no key")):
        explanation = rec.explain_recommendation(user, song)
    assert song.title in explanation
