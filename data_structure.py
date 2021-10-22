from dataclasses import dataclass
import numpy as np
import constants as c


@dataclass
class Pitch:
    step: int
    alter: float
    octave: int

    def encode(self) -> int:
        if self.alter is not None:
            return self.octave * 12 + self.step + round(self.alter)
        else:
            return self.octave * 12 + self.step


@dataclass
class Note:
    instrument: int
    staff: int
    is_prolonged: bool
    duration: int
    position: int

    def encode(self):
        return 0


@dataclass
class Rest(Note):
    pass


@dataclass
class PitchedNote(Note):
    pitch: Pitch

    def encode(self):
        return self.pitch.encode()


@dataclass
class Grace(PitchedNote):
    pass


@dataclass
class Moment:
    notes: list[Note]
    location: int

    def encode(self):
        encoded = [0] * c.INPUT_MAX_DEPTH
        encoded[0] = self.location
        index = 0
        for note in self.notes:
            index += 1
            if index == c.INPUT_MAX_DEPTH:
                break
            encoded_note = note.encode()
            encoded[index] = encoded_note
        return np.array(encoded)


@dataclass
class Measure:
    moments: list[Moment]

    def encode(self):
        encoded = []
        for moment in self.moments:
            encoded_moment = moment.encode()
            encoded.append(encoded_moment)
        return np.array(encoded)


@dataclass
class ScorePart:
    piece_name: str
    measures: list[Measure]

    def encode(self):
        encoded = []
        for measure in self.measures:
            encoded_measure = measure.encode()
            encoded.append(encoded_measure)
        only_non_empty_measures = list(filter(lambda m: m.shape[0] != 0, encoded))
        if not only_non_empty_measures:
            return np.zeros((c.INPUT_MAX_WIDTH, c.INPUT_MAX_DEPTH))
        else:
            return np.concatenate(only_non_empty_measures)
