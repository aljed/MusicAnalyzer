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
            encoded = encoded + encoded_moment
        return np.array(encoded)


@dataclass
class ScorePart:
    measures: list[Measure]

    def encode(self):
        encoded = []
        for measure in self.measures:
            encoded_measure = measure.encode()
            encoded = encoded + encoded_measure
        padding = c.INPUT_MAX_WIDTH - len(encoded) % c.INPUT_MAX_WIDTH
        if padding != c.INPUT_MAX_WIDTH:
            encoded += [0] * c.INPUT_MAX_DEPTH
        return np.array(encoded)
