"""
Semantle game, originally from https://github.com/EthanJantz/semantle/tree/main
"""

import os
import gensim
import gensim.downloader
from gensim.models import KeyedVectors
from typing import cast

ROOT_DIR = os.path.abspath(".")


class Semantle:
    def __init__(self):
        # Load Google News Word2Vec model
        print("Loading Google News Word2Vec model...")
        self.model: KeyedVectors = cast(
            KeyedVectors, gensim.downloader.load("word2vec-google-news-300")
        )

        self.word_of_the_day = "medical"  # Changed from "bart" to a more common word
        self.guesses_dict = {}
        self.guesses_in_order = []
        self.endgame = False

    def player_guess(self) -> str:
        """
        Handles the user input for a guess.
        """
        guess = input("Guess: ")
        cleaned_guess = guess.lower()
        return cleaned_guess

    def take_turn(self) -> None:
        """
        Takes in the user's guess, updates the game log, and gives the
        player feedback on their guess.
        """
        # Use walrus operator (:=) to assign and evaluate current_guess in one expression
        # The loop continues while TWO conditions are met:
        # 1. current_guess := self.player_guess() - assigns user input to current_guess (always truthy for non-empty strings)
        # 2. self.check_guess(current_guess) is None - checks if the guess is invalid (not in vocabulary)
        #
        # The loop will exit when check_guess returns a valid similarity score (not None),
        # meaning the guessed word exists in the model's vocabulary
        while (current_guess := self.player_guess()) and self.check_guess(
            current_guess
        ) is None:
            pass  # Empty loop body - all logic is in the condition

        # At this point, current_guess is guaranteed to be valid and have a similarity score
        similarity_of_current_guess = self.check_guess(current_guess)

        # Store the guess and its similarity score
        self.guesses_dict[current_guess] = similarity_of_current_guess
        self.guesses_in_order.append(current_guess)
        print(f"{current_guess}: {similarity_of_current_guess}")
        self.update_game_state(current_guess)

    def play_game(self):
        while not self.endgame:
            self.take_turn()
    
    def check_guess(self, guess) -> float | None:
        """
        Calculates the similarity of the current guess if the guessed word
        is in the vocabulary of the game model.
        """
        try:
            similarity_of_current_guess = self.model.similarity(
                guess, self.word_of_the_day
            )
            return similarity_of_current_guess
        except Exception as _:
            return None

    def update_game_state(self, current_guess) -> None:
        """
        Sets the game state to end if the user guesses the correct word.
        """
        
        if current_guess == self.word_of_the_day:
            self.endgame = True
            print("You guessed the correct word!")
        else:
            self.endgame = False