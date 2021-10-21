from unittest import TestCase

import data_preparator
import data_structure as ds


class Test(TestCase):
    def test_get_input_data(self):
        score_parts = self.get_test_score_parts()

        input_data = data_preparator.get_input_data(score_parts)

        print(input_data)

    def get_test_score_parts(self):
        return [ds.ScorePart(self.get_test_measures(1)),
                ds.ScorePart(self.get_test_measures(2))]

    def get_test_measures(self, part_number):
        return [ds.Measure(self.get_test_moments(part_number * 10 + 1)),
                ds.Measure(self.get_test_moments(part_number * 10 + 2)),
                ds.Measure(self.get_test_moments(part_number * 10 + 3)),
                ds.Measure(self.get_test_moments(part_number * 10 + 4))]

    def get_test_moments(self, seed):
        moments_list = []
        for i in (1, 5):
            moments_list.append(ds.Moment(self.get_test_notes(seed * i), i))
        return moments_list

    def get_test_notes(self, seed):
        notes_list = []
        for i in (1, 5):
            notes_list.append(self.get_test_note(seed * i))
        return notes_list

    def get_test_note(self, seed):
        return ds.PitchedNote(seed, seed + 1, False, seed + 2, seed, ds.Pitch(seed + 1, seed + 3, seed + 5))
