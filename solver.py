import random
import numpy as np
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
        rows, cols = np.triu_indices(len(ordered), k=1)
        normals = word_vecs[rows] - word_vecs[cols]  # (n_constraints, 300)

        # Score every candidate against every constraint simultaneously
        candidate_vecs = self.vectors[candidate_mask]  # (n_cand, 300)
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
            if similarity is None:
                self.candidates.remove(guess_idx)
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


if __name__ == "__main__":
    game = Semantle()
    solver = HalfspaceSolver(game)
    solver.solve()
