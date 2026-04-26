"""
Applied AI Music System — entry point.

Run from the project root:
    python main.py

Requires ANTHROPIC_API_KEY env var for RAG-enriched explanations.
Without it, explain_recommendation() falls back to a template string.
"""
from pathlib import Path

from src.recommender import load_songs, Song, UserProfile, Recommender
from src.evaluator import compute_metrics, detect_bias

WIDTH = 62


def _header(label: str) -> None:
    print("\n" + "=" * WIDTH)
    print(f"  {label}")
    print("=" * WIDTH)


def _section(label: str) -> None:
    print(f"\n  -- {label} --")


def _recommendation(rank: int, song: Song, score: float, explanation: str) -> None:
    bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
    print(f"\n  #{rank}  {song.title} — {song.artist}")
    print(f"       Genre : {song.genre:<14}  Mood : {song.mood}")
    print(f"       Score : {score:.2f}  [{bar}]")
    # wrap explanation across lines
    sentences = [s.strip() for s in explanation.replace(". ", ".\n").splitlines() if s.strip()]
    for i, sentence in enumerate(sentences):
        prefix = "       Why   : " if i == 0 else "               "
        print(f"{prefix}{sentence}")
    print("  " + "-" * (WIDTH - 2))


def main() -> None:
    data_path = Path(__file__).parent / "data" / "songs.csv"
    songs_data = load_songs(str(data_path))
    all_songs = [Song(**s) for s in songs_data]
    rec = Recommender(all_songs)

    profiles = [
        ("Profile A: Deep Intense Rock", UserProfile(
            favorite_genre="rock", favorite_mood="intense",
            target_energy=0.90, likes_acoustic=False,
        )),
        ("Profile B: High-Energy Pop", UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.85, likes_acoustic=False,
        )),
        ("Profile C: Chill Lofi", UserProfile(
            favorite_genre="lofi", favorite_mood="chill",
            target_energy=0.38, likes_acoustic=True,
        )),
    ]

    for label, user in profiles:
        _header(f"{label} — Top 5")

        top5 = rec.score_all(user)[:5]
        for rank, (song, score) in enumerate(top5, start=1):
            explanation = rec.explain_recommendation(user, song)
            _recommendation(rank, song, score, explanation)

        _section("Evaluation Metrics")
        metrics = compute_metrics(top5, user)
        print(metrics.report())

        _section("Bias Report")
        bias = detect_bias(top5, user, all_songs)
        print(bias.summary())


if __name__ == "__main__":
    main()
