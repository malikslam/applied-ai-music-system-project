# Applied AI Music System

## Original Project

This project extends **VibeMatcher 1.0**, built in [Module 3 of AI 110](https://github.com/malikslam/ai110-module3show-musicrecommendersimulation-starter). VibeMatcher was a content-based music recommender that scored songs by comparing audio features (energy, mood, acousticness, valence, danceability, tempo) against a user's taste profile using a weighted similarity formula. It demonstrated that transparent, rule-based scoring could produce recommendations that *feel* personalized — and surfaced how weight choices like `mood = 2.0` create real, measurable bias by overriding genre preference for users who want to explore adjacent sounds.

---

## Title and Summary

**Applied AI Music System** extends VibeMatcher into a complete applied AI pipeline with three new layers:

1. **RAG-enriched explanations** — `explain_recommendation()` retrieves genre, mood, and feature descriptions from a local knowledge base and passes them to Claude (Haiku) alongside the score breakdown, producing natural-language explanations grounded in music theory rather than raw numbers.
2. **Bias detection** — four automated checks flag Mood Dominance, Genre Filter Bubbles, Catalog Coverage Gaps, and Score Cliffs on every set of results.
3. **Evaluation metrics** — `compute_metrics()` measures precision@k, genre/mood precision, genre coverage (a diversity signal), mean score, and score standard deviation after each recommendation run.

This matters because "showing reasons" is not the same as being transparent. The original VibeMatcher showed *what* matched, but a user reading `mood matched (happy)` still couldn't see that mood's 2× weight was silently overriding their genre preference. The new system names that problem explicitly every time it happens.

---

## Architecture Overview

```
main.py
  │
  ├── src/recommender.py          Core scoring + Recommender class
  │     ├── score_song()          Weighted similarity (unchanged from VibeMatcher)
  │     ├── Recommender.score_all()   Scores all songs, returns sorted (Song, float) pairs
  │     ├── Recommender.recommend()   Returns top-k Song objects
  │     └── Recommender.explain_recommendation()
  │           │
  │           ├── score_song()          Get per-feature breakdown + reasons
  │           ├── src/knowledge_base.py
  │           │     └── retrieve_context(genre, mood, features)
  │           │           Returns genre doc + mood doc + up to 2 feature docs
  │           │           (this is the RAG retrieval step)
  │           │
  │           ├── Claude API (claude-haiku-4-5)
  │           │     Prompt = user profile + song + score breakdown + retrieved context
  │           │     Response = 2–3 sentence natural-language explanation
  │           │
  │           └── Validation
  │                 If score < 0.4 and LLM says "perfect match" → append correction note
  │                 If API key missing or call fails → fallback template explanation
  │
  └── src/evaluator.py
        ├── compute_metrics(results, user)   → EvalMetrics
        └── detect_bias(results, user, all_songs) → BiasReport
              Checks: Mood Dominance · Genre Filter Bubble
                      Catalog Coverage Gap · Score Cliff
```

The knowledge base (`src/knowledge_base.py`) is a local dictionary — no vector database or embedding model. Retrieval is deterministic: look up the song's genre, mood, and top-scoring numeric features by key. This keeps the RAG step inspectable and free of external dependencies.

---

## Setup Instructions

**1. Clone or copy this folder**

```bash
cd applied-ai-music-system-project
```

**2. Install dependencies**

```bash
pip3 install -r requirements.txt
```

**3. Set your Anthropic API key** *(required for RAG-enriched explanations)*

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Without this key, `explain_recommendation()` falls back to a template string automatically — the rest of the system (scoring, metrics, bias detection) works without it.

**4. Run the recommender**

```bash
python3 main.py
```

**5. Run the test suite**

```bash
python3 -m pytest tests/ -v
```

Expected: **26 passed**.

---

## Sample Interactions

### Profile A — Deep Intense Rock

**Input profile:**
```python
UserProfile(favorite_genre="rock", favorite_mood="intense", target_energy=0.90, likes_acoustic=False)
```

**Output (top 3 of 5):**
```
#1  Storm Runner — Voltline
    Genre : rock            Mood : intense
    Score : 0.98  [###################-]
    Why   : Storm Runner by Voltline scored 0.98.
            Match signals: genre matched (rock), mood matched (intense),
            energy is a close match (0.91), acousticness is a close match (0.1).

#2  Gym Hero — Max Pulse
    Genre : pop             Mood : intense
    Score : 0.80  [################----]
    Why   : Gym Hero by Max Pulse scored 0.80.
            Match signals: mood matched (intense), energy is a close match (0.93),
            acousticness is a close match (0.05).

#3  Sunrise City — Neon Echo
    Genre : pop             Mood : happy
    Score : 0.47  [#########-----------]
```

**Evaluation Metrics:**
```
precision@k (genre+mood exact): 0.20
genre precision:                0.20
mood precision:                 0.40
genre coverage (diversity):     0.80
mean score:                     0.62
score std dev:                  0.26
```

**Bias Report:**
```
[LOW] Score Cliff: Score range 0.98–0.42 (gap=0.57). Lower-ranked songs are weak alternatives.
```

**What this shows:** The system correctly surfaces Storm Runner as the dominant match. The low `precision@k` of 0.20 (only 1 of 5 results matches both genre and mood) reveals there is only one rock/intense song in the 10-song catalog — a coverage problem the bias detector does *not* catch here because the combo exists, but the metrics make the scarcity visible.

---

### Profile B — High-Energy Pop

**Input profile:**
```python
UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.85, likes_acoustic=False)
```

**Output (top 3 of 5):**
```
#1  Sunrise City — Neon Echo
    Genre : pop             Mood : happy
    Score : 0.98  [###################-]

#2  Rooftop Lights — Indigo Parade
    Genre : indie pop       Mood : happy
    Score : 0.76  [###############-----]
    Why   : Rooftop Lights by Indigo Parade scored 0.76.
            Match signals: mood matched (happy), energy is a close match (0.76).

#3  Gym Hero — Max Pulse
    Genre : pop             Mood : intense
    Score : 0.62  [############--------]
    Why   : Gym Hero by Max Pulse scored 0.62.
            Match signals: genre matched (pop), energy is a close match (0.93),
            acousticness is a close match (0.05).
```

**Evaluation Metrics:**
```
precision@k (genre+mood exact): 0.20
genre precision:                0.40
mood precision:                 0.40
genre coverage (diversity):     0.80
mean score:                     0.66
score std dev:                  0.22
```

**Bias Report:**
```
[LOW] Score Cliff: Score range 0.98–0.46 (gap=0.53). Lower-ranked songs are weak alternatives.
```

**What this shows:** This is the classic VibeMatcher bias case. Rooftop Lights (indie pop/happy) ranks #2 above Gym Hero (pop/intense) at #3 — an exact genre match loses to a mood match because mood is weighted 2×. The metrics name it: `genre_precision = 0.40`, `mood_precision = 0.40`, and no Mood Dominance flag fires because pop *does* appear in the top 3 — yet the user still sees indie pop ranked above their preferred genre's own songs. The score cliff confirms the drop-off after rank 1 is steep.

---

### Profile C — Chill Lofi

**Input profile:**
```python
UserProfile(favorite_genre="lofi", favorite_mood="chill", target_energy=0.38, likes_acoustic=True)
```

**Output (top 3 of 5):**
```
#1  Library Rain — Paper Lanterns
    Genre : lofi            Mood : chill
    Score : 0.99  [###################-]

#2  Midnight Coding — LoRoom
    Genre : lofi            Mood : chill
    Score : 0.96  [###################-]

#3  Spacewalk Thoughts — Orbit Bloom
    Genre : ambient         Mood : chill
    Score : 0.79  [###############-----]
```

**Evaluation Metrics:**
```
precision@k (genre+mood exact): 0.40
genre precision:                0.60
mood precision:                 0.60
genre coverage (diversity):     0.60
mean score:                     0.77
score std dev:                  0.21
```

**Bias Report:**
```
[LOW] Score Cliff: Score range 0.99–0.49 (gap=0.50). Lower-ranked songs are weak alternatives.
```

**What this shows:** Lofi performs best here — highest `precision@k` (0.40) and highest `mean_score` (0.77) of the three profiles. The lower `genre_coverage` (0.60) flags that results cluster around lofi and ambient, which is a mild echo-chamber signal. The lofi/chill catalog is the most represented combo, so this profile gets "rewarded" by the data distribution — a data-level bias the metrics surface.

---

## Design Decisions

### Why a local dictionary for RAG, not a vector database?

The knowledge base is 14 short documents (7 genres + 6 moods + 5 features). Spinning up a vector database or calling an embeddings API for this would add latency, cost, and a dependency without meaningfully improving retrieval quality. The lookup is deterministic: a song's genre key either exists in `GENRE_DOCS` or it doesn't. This makes the retrieval step inspectable — you can read exactly what context Claude receives.

**Trade-off:** This doesn't generalize. A real music service with thousands of genres would need semantic retrieval. For 10 songs and a fixed catalog, dictionary lookup is more transparent and equally accurate.

### Why Claude Haiku for explanations?

Explanations are 2–3 sentences with a well-structured prompt and no ambiguity about the expected output. Haiku is fast enough that it doesn't slow the CLI output noticeably and costs less per call than Sonnet or Opus. The task doesn't require deep reasoning — it requires faithful summarization of the score breakdown enriched by the retrieved context.

**Trade-off:** Haiku occasionally produces generic phrasing. Sonnet would produce richer explanations but at 5× the cost for a feature that already has a working template fallback.

### Why separate `compute_metrics()` and `detect_bias()` instead of one function?

They answer different questions. Metrics quantify *how well* the results match the user — they're numeric and comparable across profiles. Bias detection checks for *structural problems* in the results — it fires flags only when something is wrong. Keeping them separate means you can show metrics on every run and only surface bias flags when they're warranted.

### Why keep the functional interface (`recommend_songs`, `score_song`)?

The original VibeMatcher used a functional interface. Removing it would break any code that imported from the original `recommender.py`. The OOP `Recommender` class layers on top of the same underlying `score_song()` logic — no duplication, just a clean public API for the new features.

### Why validate the LLM output?

LLMs occasionally overclaim. A song with a 0.32 match score could receive an explanation calling it "an excellent match." The validation step checks for this exact pattern and appends a correction note. It's a lightweight guardrail — not a full re-ranking or critique loop — but it prevents the explanation from actively misleading the user.

---

## Testing Summary

The system uses four overlapping reliability mechanisms:

| Mechanism | Where | What it covers |
|---|---|---|
| **Automated tests** | `tests/` (26 tests, all pass) | Scoring correctness, bias flags, metrics edge cases, fallback path |
| **Confidence scoring** | `score_song()` → 0–1 score; `EvalMetrics.mean_score` | Per-song match strength; per-profile average reliability |
| **Logging + error handling** | `src/recommender.py` | `[WARNING]` logged with song title, score, and exception whenever the Claude call fails and the fallback template is used |
| **Human evaluation** | Sample Interactions section above | Three profiles manually reviewed against expected ranking behavior |

**Quick summary:** 26/26 tests passed. Confidence scores on the three test profiles averaged 0.62–0.77 (Profile C highest at 0.77, Profile A lowest at 0.62). Accuracy improved after adding the LLM validation guardrail, which flags over-confident explanations when the match score is below 0.40. When no API key is set, every call triggers a logged warning:

```
[WARNING] src.recommender: explain_recommendation falling back to template
for 'Storm Runner' (score=0.98): ANTHROPIC_API_KEY not set — falling back to template.
```

**Test breakdown:**

| Test file | Tests | What they cover |
|---|---|---|
| `tests/test_evaluator.py` | 20 | `compute_metrics()` edge cases, all 4 bias flags firing and not-firing, `BiasReport` methods |
| `tests/test_recommender.py` | 6 | `recommend()` count/sort/k, `score_all()` order, `explain_recommendation()` non-empty, fallback mock |

**What worked:**

- All four bias flags behave correctly: each fires under the exact conditions it should and stays silent when conditions aren't met. The mock test in `test_explain_recommendation_fallback_contains_title` confirmed the fallback path works without an API key by patching `_generate_explanation` to raise.
- `compute_metrics()` handles edge cases cleanly — empty results return zeroed metrics, partial matches calculate correctly.
- The test for `score_std < 0.05` as a filter-bubble signal works: two lofi/chill songs with scores 0.99 and 0.97 produce a standard deviation of 0.014, well below the threshold.

**What didn't work initially:**

- The `recommend_songs()` functional interface had a bug in the first draft: unpacking `(song, score, reasons)` from `score_song()` which returns `(score, reasons)` — not three values. Fixed by unpacking the tuple correctly: `(song, *score_song(...))` and then manually joining reasons.
- The `_features_from_reasons()` helper initially missed `"tempo"` → `"tempo_bpm"` mapping because `score_song()` writes `"tempo is a close match"` (not `"tempo_bpm"`). Fixed by adding the `_REASON_FIRST_WORD_TO_FEATURE` mapping dictionary in `knowledge_base.py`.

**What I learned from testing:**

Writing tests for bias detection before writing the full detection logic forced precision about what "60% threshold" and "top-3" actually mean at edge cases. For example: does a single-song result list that is 100% one genre trigger the filter bubble? Yes — `1/1 > 0.6` is `True`. Does the mood dominance check look at the top-3 or the full top-k? The tests locked in the decision: top-3, because that's what a user sees first.

---

## Reflection

### What this project taught me about AI

The biggest lesson from VibeMatcher was that simple algorithms *feel* intelligent. The biggest lesson from this extension is that adding language model output doesn't solve that problem — it can make it worse. When the LLM produces a fluent, confident sentence like "This track's intense mood and driving energy make it a strong match for your profile," the user has even less reason to ask *why* Gym Hero ranked above Storm Runner. Good explanations can function as propaganda for the model's choices.

The bias detection layer is a counter-weight to this. It doesn't stop the recommender from making biased choices — nothing in this system does — but it names the patterns explicitly: "Mood weight (2.0) may be overriding genre preference (1.0)." That forces the output to acknowledge a limitation the explanation might otherwise obscure.

### On RAG as a design pattern

RAG felt most valuable here not for accuracy but for *grounding*. The LLM without context could still generate a plausible explanation for any song — it has training data about genres and moods. But that explanation might use knowledge that contradicts the actual scoring logic (for example, claiming a song was recommended for its valence when valence wasn't in the user's profile dict). Injecting the score breakdown and knowledge-base context constrains the model to explain *this run*, not the general concept of recommendation.

The retrieval step being deterministic — a dictionary lookup, not semantic search — was a conscious choice. It's easy to audit. If Claude says something unexpected about a rock song's characteristics, I can read the exact text that was retrieved and trace where the explanation came from.

### On evaluation as a first-class concern

I built VibeMatcher first and evaluated it second, by eye, running three profiles and seeing which results surprised me. The extension reverses that priority: evaluation runs automatically on every output. This changed how I thought about the system. Once you can measure `genre_coverage`, you start asking "what's a good coverage score?" and "should the system ever trade precision for diversity?" Those are design questions that only become visible when you have numbers to argue about.

The most honest thing the metrics surface: even the best-performing profile (lofi/chill, `precision@k = 0.40`) gets fewer than half its top-5 results exactly right on both genre and mood. For a 10-song catalog where 3 songs are lofi/chill, that means the catalog itself is the binding constraint — not the algorithm, not the weights, not the LLM. Evaluating the system made the data problem undeniable.
