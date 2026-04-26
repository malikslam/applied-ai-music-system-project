"""
Evaluation metrics and bias detection for the music recommender.

compute_metrics()  — precision@k, genre/mood precision, coverage, score stats
detect_bias()      — four bias checks returning a BiasReport
"""
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple

from .recommender import Song, UserProfile


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    precision_at_k: float   # fraction of top-k matching genre AND mood exactly
    genre_precision: float  # fraction matching user's preferred genre
    mood_precision: float   # fraction matching user's preferred mood
    genre_coverage: float   # unique genres in results / k  (diversity signal)
    mean_score: float       # average recommendation score
    score_std: float        # std dev of scores (low = filter bubble)

    def report(self) -> str:
        bubble_note = "  <- low diversity" if self.score_std < 0.05 else ""
        lines = [
            f"  precision@k (genre+mood exact): {self.precision_at_k:.2f}",
            f"  genre precision:                {self.genre_precision:.2f}",
            f"  mood precision:                 {self.mood_precision:.2f}",
            f"  genre coverage (diversity):     {self.genre_coverage:.2f}",
            f"  mean score:                     {self.mean_score:.2f}",
            f"  score std dev:                  {self.score_std:.2f}{bubble_note}",
        ]
        return "\n".join(lines)


@dataclass
class BiasFlag:
    name: str
    description: str
    severity: str  # "low" | "medium" | "high"


@dataclass
class BiasReport:
    flags: List[BiasFlag] = field(default_factory=list)

    def has_bias(self) -> bool:
        return bool(self.flags)

    def summary(self) -> str:
        if not self.flags:
            return "  No bias flags detected."
        return "\n".join(
            f"  [{f.severity.upper()}] {f.name}: {f.description}"
            for f in self.flags
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    results: List[Tuple[Song, float]],
    user: UserProfile,
) -> EvalMetrics:
    """Compute evaluation metrics for a ranked list of (song, score) pairs."""
    k = len(results)
    if k == 0:
        return EvalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    songs = [s for s, _ in results]
    scores = [sc for _, sc in results]

    genre_hits = sum(1 for s in songs if s.genre == user.favorite_genre)
    mood_hits = sum(1 for s in songs if s.mood == user.favorite_mood)
    exact_hits = sum(
        1 for s in songs
        if s.genre == user.favorite_genre and s.mood == user.favorite_mood
    )
    unique_genres = len({s.genre for s in songs})

    return EvalMetrics(
        precision_at_k=exact_hits / k,
        genre_precision=genre_hits / k,
        mood_precision=mood_hits / k,
        genre_coverage=unique_genres / k,
        mean_score=sum(scores) / k,
        score_std=statistics.stdev(scores) if k > 1 else 0.0,
    )


# ---------------------------------------------------------------------------
# Bias detection
# ---------------------------------------------------------------------------

def detect_bias(
    results: List[Tuple[Song, float]],
    user: UserProfile,
    all_songs: List[Song],
) -> BiasReport:
    """
    Run four bias checks on the recommendation results.

    Checks:
      1. Mood Dominance      — user's preferred genre absent from top-3
      2. Genre Filter Bubble — >60% of results share the same genre
      3. Catalog Coverage Gap — no song in catalog has user's exact genre+mood combo
      4. Score Cliff         — score gap between rank-1 and rank-k exceeds 0.4
    """
    report = BiasReport()
    songs = [s for s, _ in results]
    scores = [sc for _, sc in results]
    k = len(songs)

    # 1. Mood dominance: user's preferred genre missing from top-3
    top3_genres = {s.genre for s in songs[:min(3, k)]}
    if user.favorite_genre not in top3_genres:
        report.flags.append(BiasFlag(
            name="Mood Dominance",
            description=(
                f"Genre '{user.favorite_genre}' is absent from the top-3 results. "
                f"Mood weight (2.0) may be overriding genre preference (1.0)."
            ),
            severity="high",
        ))

    # 2. Genre filter bubble: more than 60% of results share a single genre
    if k > 0:
        genre_counts = Counter(s.genre for s in songs)
        top_genre, top_count = genre_counts.most_common(1)[0]
        if top_count / k > 0.6:
            report.flags.append(BiasFlag(
                name="Genre Filter Bubble",
                description=(
                    f"{top_count}/{k} results are '{top_genre}' — "
                    f"results may form a genre echo chamber."
                ),
                severity="medium",
            ))

    # 3. Catalog coverage gap: no song exists with user's exact genre+mood combo
    catalog_combos = {(s.genre, s.mood) for s in all_songs}
    if (user.favorite_genre, user.favorite_mood) not in catalog_combos:
        report.flags.append(BiasFlag(
            name="Catalog Coverage Gap",
            description=(
                f"No song in the catalog has genre='{user.favorite_genre}' "
                f"AND mood='{user.favorite_mood}'. System is forced to approximate."
            ),
            severity="high",
        ))

    # 4. Score cliff: large gap between rank-1 and rank-k leaves no real alternatives
    if len(scores) >= 2 and (scores[0] - scores[-1]) > 0.4:
        report.flags.append(BiasFlag(
            name="Score Cliff",
            description=(
                f"Score range {scores[0]:.2f}–{scores[-1]:.2f} "
                f"(gap={scores[0] - scores[-1]:.2f}). "
                f"Lower-ranked songs are weak alternatives."
            ),
            severity="low",
        ))

    return report
