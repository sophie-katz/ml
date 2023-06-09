from typing import (
    List,
    Union,
    Optional,
    Dict,
    Set,
    Tuple,
    cast,
    Collection,
    Callable,
)
import random
import json
import abc
import dataclasses
import enum
import functools
import itertools
import pandas as pd
from ipywidgets import IntProgress
from IPython.display import display

WORD_LENGTH = 5
TURN_COUNT = 6


@enum.unique
class Color(enum.Enum):
    GREY = 0
    YELLOW = 1
    GREEN = 2


@dataclasses.dataclass
class WordMask(object):
    global_included_letters: Set[str] = dataclasses.field(default_factory=set)
    global_excluded_letters: Set[str] = dataclasses.field(default_factory=set)
    positional_known_letters: Dict[int, str] = dataclasses.field(default_factory=dict)
    positional_excluded_letters: Dict[int, Set[str]] = dataclasses.field(
        default_factory=dict
    )

    def matches(self, word: str) -> bool:
        assert len(word) == WORD_LENGTH

        for position, letter in self.positional_known_letters.items():
            assert position >= 0
            assert position < WORD_LENGTH
            assert len(letter) == 1
            if word[position] != letter:
                return False

        for position, letters in self.positional_excluded_letters.items():
            assert position >= 0
            assert position < WORD_LENGTH
            assert not any(len(letter) != 1 for letter in letters)
            if word[position] in letters:
                return False

        for letter in self.global_included_letters:
            assert len(letter) == 1
            if not letter in word:
                return False

        for letter in self.global_excluded_letters:
            assert len(letter) == 1
            if letter in word:
                return False

        return True

    def apply_color(self, position: int, letter: str, color: Color) -> None:
        assert position >= 0
        assert position < WORD_LENGTH
        assert len(letter) == 1

        if color == Color.GREY:
            self.global_excluded_letters.add(letter)
            if not position in self.positional_excluded_letters:
                self.positional_excluded_letters[position] = set()
            self.positional_excluded_letters[position].add(letter)
        elif color == Color.YELLOW:
            self.global_included_letters.add(letter)
            if not position in self.positional_excluded_letters:
                self.positional_excluded_letters[position] = set()
            self.positional_excluded_letters[position].add(letter)
        elif color == Color.GREEN:
            self.global_included_letters.add(letter)
            self.positional_known_letters[position] = letter
        else:
            raise NotImplementedError()


class WordSet(object):
    def __init__(self, *args: Union[List[str], str]) -> None:
        self.words = set(i for i in itertools.chain(*args) if len(i) == WORD_LENGTH)

    @functools.cached_property
    def letter_frequency(self) -> pd.DataFrame:
        data: Dict[str, int] = {}

        for word in self.words:
            for letter in word:
                if letter in data:
                    data[letter] += 1
                else:
                    data[letter] = 1

        df = pd.DataFrame(
            [{"letter": key, "count": value} for key, value in data.items()]
        ).sort_values("count", ascending=False)

        df["frequency"] = df["count"] / df["count"].max()

        return df

    @functools.cached_property
    def words_in_letter_frequency_sum_order(self) -> pd.DataFrame:
        # return pd.DataFrame([{"word": word, "frequency_sum": self.get_word_frequency_sum(word)} for word in self.words]) \
        #     .sort_values("frequency_sum", ascending = False)
        return sorted(
            self.words, key=functools.partial(WordSet.get_word_frequency_sum, self)
        )

    def get_word_frequency_sum(self, word: str) -> float:
        return cast(
            float,
            self.letter_frequency.loc[self.letter_frequency["letter"].isin(list(word))][
                "frequency"
            ].sum(),
        )

    def get_random_words(self, mask: WordMask = WordMask()) -> pd.DataFrame:
        data = []

        for word in self.words:
            if mask.matches(word):
                data.append({"word": word})

        random.shuffle(data)

        return pd.DataFrame(data)

    def get_random_word(self, mask: WordMask = WordMask()) -> str:
        return random.choice([word for word in self.words if mask.matches(word)])

    def get_max_letter_frequency_sum_word(self, mask: WordMask = WordMask()) -> str:
        return cast(
            str,
            next(
                word
                for word in self.words_in_letter_frequency_sum_order
                if mask.matches(word)
            ),
        )


class WordleAgentBase(abc.ABC):
    def __init__(self, word_set: WordSet) -> None:
        self.word_set = word_set
        self.word_mask = WordMask()

    @abc.abstractmethod
    def act(self) -> str:
        pass

    def learn(self, word: str, colors: Dict[int, Color]) -> None:
        for position, color in colors.items():
            self.word_mask.apply_color(position, word[position], color)


class WordleAgentRandom(WordleAgentBase):
    def __init__(self, word_set: WordSet) -> None:
        super().__init__(word_set)

    def act(self) -> str:
        return self.word_set.get_random_word(self.word_mask)


class WordleAgentLetterFrequencySum(WordleAgentBase):
    def __init__(self, word_set: WordSet) -> None:
        super().__init__(word_set)

    def act(self) -> str:
        return self.word_set.get_max_letter_frequency_sum_word(self.word_mask)


class WordleSimulation(object):
    def __init__(self, agent_factory: Callable[[], WordleAgentBase]) -> None:
        self.agent_factory = agent_factory

    def _step(
        self, agent: WordleAgentBase, word: str, turn_index: int
    ) -> Tuple[bool, str, Dict[int, Color]]:
        assert len(word) == WORD_LENGTH
        assert turn_index >= 0

        guess = agent.act()
        assert len(guess) == WORD_LENGTH

        colors = {}

        for index in range(WORD_LENGTH):
            if guess[index] == word[index]:
                colors[index] = Color.GREEN
            elif guess[index] in word:
                colors[index] = Color.YELLOW
            else:
                colors[index] = Color.GREY

        agent.learn(guess, colors)
        return guess == word, guess, colors

    def run(
        self, word: str
    ) -> Tuple[str, bool, List[Tuple[bool, str, Dict[int, Color]]]]:
        assert len(word) == WORD_LENGTH

        results_game = []
        agent = self.agent_factory()

        for i in range(TURN_COUNT):
            results_turn = self._step(agent, word, i)
            results_game.append(results_turn)
            if results_turn[0]:
                break

        return word, results_game[-1][0], results_game

    def run_many(
        self, words: Collection[str], enable_progress_bar: bool = False
    ) -> Tuple[float, pd.DataFrame]:
        hist_data_dict: Dict[int, int] = {}
        win_count = 0

        progress_bar: Optional[IntProgress] = None

        if enable_progress_bar:
            progress_bar = IntProgress(
                min=0, max=len(words), description=f"0/{len(words)}"
            )
            display(progress_bar)  # type: ignore

        for index, word in enumerate(words):
            results_game = self.run(word)
            if results_game[1]:
                if len(results_game[2]) in hist_data_dict:
                    hist_data_dict[len(results_game[2])] += 1
                else:
                    hist_data_dict[len(results_game[2])] = 1
                win_count += 1

            if enable_progress_bar:
                cast(IntProgress, progress_bar).value += 1
                cast(IntProgress, progress_bar).description = f"{index}/{len(words)}"

        if enable_progress_bar:
            cast(IntProgress, progress_bar).value = len(words)
            cast(IntProgress, progress_bar).description = f"{len(words)}/{len(words)}"

        hist_data = []

        for turn_count, win_count in hist_data_dict.items():
            hist_data.append({"turn_count": turn_count, "win_count": win_count})

        if win_count > 0:
            df = pd.DataFrame(hist_data).sort_values("turn_count")
            df["win_frequency"] = df["win_count"] / df["win_count"].sum()
        else:
            df = pd.DataFrame()

        return win_count / len(words), df


if __name__ == "__main__":
    with open("words.json", "r") as file:
        word_set = WordSet(json.load(file))

    wordle_simulation = WordleSimulation(
        lambda: WordleAgentLetterFrequencySum(word_set)
    )

    win_ratio, hist = wordle_simulation.run_many(
        random.sample(list(word_set.words), 30)
    )

    print(hist)
    print(f"Win ratio: {win_ratio}")
