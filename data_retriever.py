import xml.etree.ElementTree as ET
import data_structure as ds
from itertools import groupby
import constants as c
import numpy as np

class XMLConverter:

    @staticmethod
    def transform_measure(xml: ET.Element) -> ds.Measure:

        notes = []
        prev_move = 0
        location = 0
        staff = 1

        for child in xml:

            if child.tag == "note":

                staff_element = child.find("staff")
                if staff_element is not None:
                    old_staff = staff
                    staff = staff_element.text
                    if staff != old_staff:
                        location = 0
                        prev_move = 0
                next_child_tag = child[0].tag
                note = None

                if next_child_tag == 'grace':
                    note = ds.Grace(1, staff, False, 0, location, ds.Pitch(0,0,0))
                elif next_child_tag == 'cue':
                    pass
                else:
                    duration = child.find('duration')
                    duration_int = int(duration.text)
                    if next_child_tag == 'chord':
                        location = location - prev_move
                    if child.find('rest'):
                        note = ds.Rest(1, staff, False, duration_int, location)
                    elif child.find('unpitched'):
                        pass
                    elif child.find('pitch'):
                        pitch_xml = child.find('pitch')
                        step = c.STEP[pitch_xml.find('step').text]
                        octave = int(pitch_xml.find('octave').text)
                        alter_element = pitch_xml.find('alter')
                        alter = None
                        if alter_element:
                            alter = int(alter_element.text)
                        pitch = ds.Pitch(step, alter, octave)
                        note = ds.PitchedNote(1, staff, False, duration_int, location, pitch)

                if note:
                    notes.append(note)
                    prev_move = note.duration
                    location = location + note.duration

        return ds.Measure(XMLConverter.organize_measure(notes))

    @staticmethod
    def organize_measure(notes: list[ds.Note]) -> list[ds.Moment]:
        moments = []
        notes.sort(key=lambda note: note.position)
        for key, group in groupby(notes, key=lambda note: note.position):
            moments.append(ds.Moment(list(group), key))
        return moments

