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
    constraint vectors instead of the game's word2vec model. Since the two models
    disagree on fine-grained ordering, hard filtering would be too aggressive —
    so instead we score each candidate by how many constraints it satisfies and
    keep the top scorers each round.
    """

    _WORD_RE = re.compile(r'^[a-z]+$')

    def __init__(self, semantle: Semantle, score_threshold: float = 0.85):

        self.semantle = semantle
        self.score_threshold = score_threshold  # keep candidates scoring >= this fraction of the max

        # game model — used only to look up guess vectors by name for constraint normals
        game_model = semantle.model
        self.game_vocab = game_model.index_to_key
        self.game_word_to_idx = game_model.key_to_index
        self.game_vectors = game_model.get_normed_vectors()

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
        self.candidates = list(range(len(self.vocab)))

        target = semantle.word_of_the_day
        self.target_idx = self.word_to_idx.get(target)  # None if target not in st vocab

    def _update_candidates(self):
        # Ordering comes from game model similarities, but constraint vectors come
        # from sentence-transformer — so all dot products stay in the same vector space.
        # Use all pairs, with binary counting of satisfied constraints.
        ordered = sorted(self.guesses, key=lambda x: -x[1])
        word_vecs = np.array([self.vectors[self.word_to_idx[w]] for w, _ in ordered])

        rows, cols = np.triu_indices(len(ordered), k=1)
        normals = word_vecs[rows] - word_vecs[cols]  # (n_constraints, 384)

        # Score candidates by how many constraints they satisfy
        candidate_vecs = self.vectors[self.candidates]
        scores = (normals @ candidate_vecs.T > 0).sum(axis=0)

        # Keep candidates within score_threshold of the best score
        max_score = scores.max()
        if max_score > 0:
            cutoff = int(max_score * self.score_threshold)
            self.candidates = [idx for idx, s in zip(self.candidates, scores) if s >= cutoff]

    def solve(self):
        print(f"Target: '{self.semantle.word_of_the_day}'")

        round_num = 0

        while len(self.candidates) > 1:
            round_num += 1
            guess_idx = random.choice(self.candidates)
            guess = self.vocab[guess_idx]

            similarity = self.semantle.check_guess(guess)
            # Remove the guessed word regardless — re-guessing adds no new constraints
            self.candidates.remove(guess_idx)
            if similarity is None:
                continue

            self.guesses.append((guess, similarity))

            if len(self.guesses) >= 2:
                before = len(self.candidates)
                target_was_present = self.target_idx in self.candidates
                self._update_candidates()
                after = len(self.candidates)
                target_now_present = self.target_idx in self.candidates
                eliminated_marker = "  *** target eliminated ***" if target_was_present and not target_now_present else ""
                print(f"Round {round_num:3d}: '{guess}' (sim={similarity:.4f})  "
                      f"{before:>8,} -> {after:>8,} candidates{eliminated_marker}")
            else:
                print(f"Round {round_num:3d}: '{guess}' (sim={similarity:.4f})  "
                      f"(need 2 guesses for constraints)")

            if guess == self.semantle.word_of_the_day:
                print(f"\nSolved! The word was '{guess}'")
                return guess

        if len(self.candidates) == 1:
            answer = self.vocab[self.candidates[0]]
            correct = self.semantle.check_guess(answer) == 1.0
            result = "correct!" if correct else f"wrong (target was '{self.semantle.word_of_the_day}')"
            print(f"\nBest guess: '{answer}' after {round_num} guesses — {result}")
            return answer

        raise ValueError("No candidates remaining — something went wrong")


if __name__ == "__main__":
    cross = "--cross" in sys.argv

    game = Semantle()
    solver = CrossModelSolver(game) if cross else HalfspaceSolver(game)
    solver.solve()
