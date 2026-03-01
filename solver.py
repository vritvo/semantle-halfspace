import re
import random
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from semantle.semantle import Semantle


class HalfspaceSolver:
    def __init__(self, semantle: Semantle):
        self.semantle = semantle
        model = semantle.model
        self.vocab = model.index_to_key
        self.word_to_idx = model.key_to_index
        self.vectors = model.get_normed_vectors()  # unit vectors, needed for cosine dot products

        self.guesses: list[tuple[str, float]] = []  # (word, similarity), unsorted
        self.candidates = list(range(len(self.vocab)))  # start with every word as a candidate

    def _update_candidates(self):
        # Key insight: if sim(w_i, target) > sim(w_j, target) for unit vectors,
        # then dot(w_i - w_j, target) > 0. Each ordered pair gives a halfspace
        # the target must lie in. We filter to candidates satisfying all of them.
        ordered = sorted(self.guesses, key=lambda x: -x[1])
        word_vecs = np.array([self.vectors[self.word_to_idx[w]] for w, _ in ordered])

        candidate_mask = np.zeros(len(self.vocab), dtype=bool)
        candidate_mask[self.candidates] = True

        # One normal vector per (i, j) pair where i ranked higher than j
        # where "normal" is the vector perpendicular to the line connecting i and j.
        rows, cols = np.triu_indices(len(ordered), k=1)
        normals = word_vecs[rows] - word_vecs[cols]  # (n_constraints, d)

        # Score every candidate against every constraint simultaneously
        candidate_vecs = self.vectors[candidate_mask]  # (n_cand, d)
        scores = normals @ candidate_vecs.T            # (n_constraints, n_cand)

        # Keep only candidates that satisfy all constraints
        still_valid = np.all(scores > 0, axis=0)
        candidate_mask[candidate_mask] = still_valid

        self.candidates = list(np.where(candidate_mask)[0])

    def solve(self):
        print(f"Starting solve for target: '{self.semantle.word_of_the_day}'")
        print(f"Vocab size: {len(self.vocab)}\n")

        round_num = 0

        while len(self.candidates) > 1:
            round_num += 1
            # Pick a random surviving candidate as the next guess
            guess_idx = random.choice(self.candidates)
            guess = self.vocab[guess_idx]

            similarity = self.semantle.check_guess(guess)
            # Remove the guessed word regardless — re-guessing adds no new constraints
            # and a duplicate guess creates a zero normal vector that wipes all candidates
            self.candidates.remove(guess_idx)
            if similarity is None:
                continue

            self.guesses.append((guess, similarity))

            if len(self.guesses) >= 2:
                before = len(self.candidates)
                self._update_candidates()
                after = len(self.candidates)
                print(f"Round {round_num:3d}: guessed '{guess}' (sim={similarity:.4f})  "
                      f"{before:>8,} -> {after:>8,} candidates")
            else:
                print(f"Round {round_num:3d}: guessed '{guess}' (sim={similarity:.4f})  "
                      f"(need 2 guesses for constraints)")

            if guess == self.semantle.word_of_the_day:
                print(f"\nSolved! The word was '{guess}' (guessed it directly)")
                return guess

        if len(self.candidates) == 1:
            answer = self.vocab[self.candidates[0]]
            print(f"\nSolved! Answer is '{answer}' after {round_num} guesses")
            return answer

        raise ValueError("No candidates remaining — something went wrong")


class CrossModelSolver:
    """
    Same halfspace idea, but uses a separate sentence-transformer model for the
    constraint vectors instead of the game's word2vec model. Instead of hard
    filtering (which eliminates the target too early when models disagree),
    uses a probabilistic approach: each constraint contributes a sigmoid
    probability, and the total score for a candidate is the sum of log-
    probabilities across all constraints. No word is ever fully eliminated.
    """

    _WORD_RE = re.compile(r'^[a-z]+$')

    def __init__(self, semantle: Semantle, steepness: float = 5.0):

        self.semantle = semantle
        self.steepness = steepness  # sigmoid steepness: higher = more like hard filtering

        # game model — not used for constraints, just for game interaction
        game_model = semantle.model
        self.game_vocab = game_model.index_to_key
        self.game_word_to_idx = game_model.key_to_index

        # constraint model — encodes the candidate vocab we'll score against
        print("Loading sentence-transformer model...")
        st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Filter to simple single-word entries present in both models
        self.vocab = [w for w in self.game_vocab if self._WORD_RE.match(w)]
        print(f"Encoding {len(self.vocab)} words through sentence-transformer...")
        self.vectors = st_model.encode(
            self.vocab, batch_size=512, show_progress_bar=True, normalize_embeddings=True
        )
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        print(f"Done. Candidate vocab: {len(self.vocab)} words\n")

        self.guesses: list[tuple[str, float]] = []

        # Log-probability for each word — starts uniform (all zeros = equal probability)
        self.log_probs = np.zeros(len(self.vocab))

        target = semantle.word_of_the_day
        self.target_idx = self.word_to_idx.get(target)  # for debug tracking

    def _log_sigmoid(self, x):
        # Computes log(sigmoid(x)) using two equivalent formulas — one for x > 0, one for x <= 0.
        # Both are the same math, rearranged so we always exponentiate a negative number
        # (which safely underflows to 0) rather than a large positive (which overflows to inf).
        return np.where(x > 0, -np.log1p(np.exp(-x)), x - np.log1p(np.exp(x)))

    def _update_log_probs(self, new_guess_word: str, new_guess_sim: float):
        # Only add constraints between the new guess and all previous guesses,
        # since previous-vs-previous constraints were already applied.
        new_vec = self.vectors[self.word_to_idx[new_guess_word]]

        prev_vecs = np.array([self.vectors[self.word_to_idx[w]] for w, _ in self.guesses[:-1]])
        prev_sims = np.array([s for _, s in self.guesses[:-1]])

        # Constraint: the higher-similarity word should be closer to the target.
        # Normal points from lower-sim toward higher-sim word.
        normals = np.where(
            (new_guess_sim > prev_sims)[:, None],
            new_vec - prev_vecs,
            prev_vecs - new_vec
        )  # (n_prev, 384)

        # Grid of dot products: one per (constraint, vocab word) pair — shape (n_prev, n_vocab)
        dot_products = normals @ self.vectors.T
        # log-sigmoid values are <= 0, so scores only decrease. Words satisfying constraints
        # lose little; words violating them lose a lot. Relative ranking is what matters.
        self.log_probs += self._log_sigmoid(self.steepness * dot_products).sum(axis=0)

    def solve(self, max_rounds: int = 500):
        print(f"Target: '{self.semantle.word_of_the_day}'")

        round_num = 0
        guessed = set()

        while round_num < max_rounds:
            round_num += 1

            # Sample the next guess proportional to posterior probability.
            # Subtract max for numerical stability before exponentiating.
            log_p = self.log_probs.copy()
            # Zero out already-guessed words so we don't repeat
            for idx in guessed:
                log_p[idx] = -np.inf

            # Convert log_probs → probabilities for sampling.
            # Shift by max first: without this, very negative values (e.g. -50000) would
            # underflow to 0 after exp(), losing all relative differences between words.
            log_p -= log_p.max()
            probs = np.exp(log_p)   # undo the log: exp(log(p)) = p
            probs /= probs.sum()    # normalize to sum to 1

            guess_idx = np.random.choice(len(self.vocab), p=probs)
            guess = self.vocab[guess_idx]
            guessed.add(guess_idx)

            similarity = self.semantle.check_guess(guess)
            if similarity is None:
                continue

            self.guesses.append((guess, similarity))

            if len(self.guesses) >= 2:
                self._update_log_probs(guess, similarity)

                # Report the target's rank in the current posterior
                target_rank = None
                if self.target_idx is not None:
                    target_rank = (self.log_probs > self.log_probs[self.target_idx]).sum() + 1

                rank_info = f"  (target rank: {target_rank})" if target_rank is not None else ""
                print(f"Round {round_num:3d}: '{guess}' (sim={similarity:.4f}){rank_info}")
            else:
                print(f"Round {round_num:3d}: '{guess}' (sim={similarity:.4f})  "
                      f"(need 2 guesses for constraints)")

            if guess == self.semantle.word_of_the_day:
                print(f"\nSolved! The word was '{guess}' after {round_num} guesses")
                return guess

        # Hit max rounds — return the highest-probability word as best guess
        best_idx = np.argmax(self.log_probs)
        best_word = self.vocab[best_idx]
        correct = best_word == self.semantle.word_of_the_day
        result = "correct!" if correct else f"wrong (target was '{self.semantle.word_of_the_day}')"
        print(f"\nHit {max_rounds} round limit. Best guess: '{best_word}' — {result}")
        return best_word


if __name__ == "__main__":
    cross = "--cross" in sys.argv

    game = Semantle()
    solver = CrossModelSolver(game) if cross else HalfspaceSolver(game)
    solver.solve()
