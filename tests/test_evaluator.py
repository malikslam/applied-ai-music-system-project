"""Tests for compute_metrics() and detect_bias() in evaluator.py."""
import pytest
from src.recommender import Song, UserProfile
from src.evaluator import BiasFlag, BiasReport, compute_metrics, detect_bias


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _catalog() -> list:
    return [
        Song(id=1, title="Rock A",  artist="X", genre="rock",  mood="intense",
             energy=0.9, tempo_bpm=150, valence=0.5, danceability=0.6, acousticness=0.1),
        Song(id=2, title="Rock B",  artist="X", genre="rock",  mood="moody",
             energy=0.7, tempo_bpm=130, valence=0.4, danceability=0.5, acousticness=0.2),
        Song(id=3, title="Pop A",   artist="Y", genre="pop",   mood="happy",
             energy=0.8, tempo_bpm=120, valence=0.9, danceability=0.8, acousticness=0.2),
        Song(id=4, title="Lofi A",  artist="Z", genre="lofi",  mood="chill",
             energy=0.4, tempo_bpm=80,  valence=0.6, danceability=0.5, acousticness=0.8),
        Song(id=5, title="Lofi B",  artist="Z", genre="lofi",  mood="chill",
             energy=0.3, tempo_bpm=75,  valence=0.5, danceability=0.4, acousticness=0.9),
    ]


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def test_metrics_perfect_genre_and_mood_match():
    catalog = _catalog()
    user = UserProfile(favorite_genre="rock", favorite_mood="intense",
                       target_energy=0.9, likes_acoustic=False)
    results = [(catalog[0], 0.99), (catalog[0], 0.90)]
    m = compute_metrics(results, user)
    assert m.precision_at_k == 1.0
    assert m.genre_precision == 1.0
    assert m.mood_precision == 1.0


def test_metrics_zero_exact_match():
    catalog = _catalog()
    user = UserProfile(favorite_genre="jazz", favorite_mood="relaxed",
                       target_energy=0.4, likes_acoustic=True)
    results = [(catalog[2], 0.60), (catalog[3], 0.55)]
    m = compute_metrics(results, user)
    assert m.precision_at_k == 0.0
    assert m.genre_precision == 0.0
    assert m.mood_precision == 0.0


def test_metrics_partial_genre_match():
    catalog = _catalog()
    user = UserProfile(favorite_genre="rock", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)
    # one rock song, one pop song
    results = [(catalog[0], 0.85), (catalog[2], 0.70)]
    m = compute_metrics(results, user)
    assert m.genre_precision == pytest.approx(0.5)
    assert m.mood_precision == pytest.approx(0.5)   # catalog[2] is pop/happy
    assert m.precision_at_k == 0.0                  # no rock+happy song


def test_metrics_genre_coverage_all_different():
    catalog = _catalog()
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)
    # 3 results from 3 different genres
    results = [(catalog[2], 0.9), (catalog[0], 0.7), (catalog[3], 0.5)]
    m = compute_metrics(results, user)
    assert m.genre_coverage == pytest.approx(1.0)


def test_metrics_genre_coverage_all_same():
    catalog = _catalog()
    user = UserProfile(favorite_genre="lofi", favorite_mood="chill",
                       target_energy=0.4, likes_acoustic=True)
    results = [(catalog[3], 0.99), (catalog[4], 0.97)]
    m = compute_metrics(results, user)
    assert m.genre_coverage == pytest.approx(0.5)   # 1 unique genre / 2 results


def test_metrics_score_std_low_signals_bubble():
    catalog = _catalog()
    user = UserProfile(favorite_genre="lofi", favorite_mood="chill",
                       target_energy=0.4, likes_acoustic=True)
    results = [(catalog[3], 0.99), (catalog[4], 0.98)]
    m = compute_metrics(results, user)
    assert m.score_std < 0.05


def test_metrics_empty_results():
    user = UserProfile(favorite_genre="rock", favorite_mood="intense",
                       target_energy=0.9, likes_acoustic=False)
    m = compute_metrics([], user)
    assert m.precision_at_k == 0.0
    assert m.mean_score == 0.0


def test_metrics_report_is_string_with_key_labels():
    catalog = _catalog()
    user = UserProfile(favorite_genre="lofi", favorite_mood="chill",
                       target_energy=0.4, likes_acoustic=True)
    results = [(catalog[3], 0.98), (catalog[4], 0.95)]
    report = compute_metrics(results, user).report()
    assert isinstance(report, str)
    assert "precision" in report.lower()
    assert "coverage" in report.lower()


# ---------------------------------------------------------------------------
# detect_bias
# ---------------------------------------------------------------------------

def test_no_mood_dominance_when_genre_in_top3():
    catalog = _catalog()
    user = UserProfile(favorite_genre="rock", favorite_mood="intense",
                       target_energy=0.9, likes_acoustic=False)
    results = [(catalog[0], 0.99), (catalog[1], 0.80), (catalog[2], 0.60)]
    bias = detect_bias(results, user, catalog)
    assert "Mood Dominance" not in [f.name for f in bias.flags]


def test_mood_dominance_flag_when_genre_absent_from_top3():
    catalog = _catalog()
    user = UserProfile(favorite_genre="jazz", favorite_mood="relaxed",
                       target_energy=0.4, likes_acoustic=True)
    results = [(catalog[2], 0.8), (catalog[3], 0.7), (catalog[4], 0.6),
               (catalog[0], 0.5), (catalog[1], 0.4)]
    bias = detect_bias(results, user, catalog)
    assert "Mood Dominance" in [f.name for f in bias.flags]


def test_genre_filter_bubble_flag():
    catalog = _catalog()
    user = UserProfile(favorite_genre="lofi", favorite_mood="chill",
                       target_energy=0.4, likes_acoustic=True)
    # 4/5 results are lofi (80% > 60% threshold)
    results = [(catalog[3], 0.99), (catalog[4], 0.98),
               (catalog[3], 0.90), (catalog[4], 0.85),
               (catalog[2], 0.50)]
    bias = detect_bias(results, user, catalog)
    assert "Genre Filter Bubble" in [f.name for f in bias.flags]


def test_no_genre_filter_bubble_when_diverse():
    catalog = _catalog()
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)
    # 5 results from 4 different genres — no single genre > 60%
    results = [(catalog[2], 0.9), (catalog[0], 0.8), (catalog[1], 0.7),
               (catalog[3], 0.6), (catalog[4], 0.5)]
    bias = detect_bias(results, user, catalog)
    assert "Genre Filter Bubble" not in [f.name for f in bias.flags]


def test_catalog_gap_flag_when_combo_missing():
    catalog = _catalog()
    user = UserProfile(favorite_genre="jazz", favorite_mood="intense",
                       target_energy=0.9, likes_acoustic=False)
    results = [(catalog[0], 0.7), (catalog[1], 0.6)]
    bias = detect_bias(results, user, catalog)
    assert "Catalog Coverage Gap" in [f.name for f in bias.flags]


def test_no_catalog_gap_when_combo_exists():
    catalog = _catalog()
    user = UserProfile(favorite_genre="rock", favorite_mood="intense",
                       target_energy=0.9, likes_acoustic=False)
    results = [(catalog[0], 0.99)]
    bias = detect_bias(results, user, catalog)
    assert "Catalog Coverage Gap" not in [f.name for f in bias.flags]


def test_score_cliff_flag():
    catalog = _catalog()
    user = UserProfile(favorite_genre="rock", favorite_mood="intense",
                       target_energy=0.9, likes_acoustic=False)
    # gap = 0.99 - 0.55 = 0.44 > 0.4
    results = [(catalog[0], 0.99), (catalog[1], 0.55)]
    bias = detect_bias(results, user, catalog)
    assert "Score Cliff" in [f.name for f in bias.flags]


def test_no_score_cliff_when_gap_small():
    catalog = _catalog()
    user = UserProfile(favorite_genre="lofi", favorite_mood="chill",
                       target_energy=0.4, likes_acoustic=True)
    results = [(catalog[3], 0.99), (catalog[4], 0.97)]
    bias = detect_bias(results, user, catalog)
    assert "Score Cliff" not in [f.name for f in bias.flags]


def test_bias_report_has_bias_true():
    bias = BiasReport(flags=[BiasFlag(name="X", description="d", severity="high")])
    assert bias.has_bias() is True


def test_bias_report_has_bias_false():
    assert BiasReport().has_bias() is False


def test_bias_report_summary_no_flags():
    assert "No bias" in BiasReport().summary()


def test_bias_report_summary_shows_severity():
    bias = BiasReport(flags=[BiasFlag(name="Test", description="desc", severity="high")])
    summary = bias.summary()
    assert "[HIGH]" in summary
    assert "Test" in summary
