import constants as c
import numpy as np
import data_structure as ds
import os
import xml.etree.ElementTree as ET
import data_retriever as dr


# Prepares input data in right shape according to hyperparameters, ready to feed it into a model
def get_input_data(score_parts: list[ds.ScorePart]) -> np.ndarray:
    encoded_score_parts = list(map(lambda score_part: score_part.encode(), score_parts))
    chunked_score_parts = np.concatenate(list(map(lambda score_part: make_chunks(score_part), encoded_score_parts)))
    return np.reshape(chunked_score_parts, (-1, c.INPUT_MAX_WIDTH, c.INPUT_MAX_DEPTH))

# Prepares input data in right shape according to hyperparameters, ready to feed it into a model
def get_input_data2(score_parts: list[ds.ScorePart]) -> np.ndarray:
    encoded_score_parts = list(map(lambda score_part: score_part.encode(), score_parts))
    chunked_score_parts = np.concatenate(list(map(lambda score_part: make_chunks(score_part), encoded_score_parts)))
    return np.reshape(chunked_score_parts, (-1, c.INPUT_MAX_WIDTH, c.INPUT_MAX_DEPTH))

# splits input data into chunks of the same size
def make_chunks(part: np.ndarray) -> np.ndarray:
    # complete array to regular value
    size = (np.shape(part))[0]
    unsplitted = part
    if size % c.INPUT_MAX_WIDTH != 0:
        number_of_columns_to_add = c.INPUT_MAX_WIDTH - size % c.INPUT_MAX_WIDTH
        to_add = np.zeros((number_of_columns_to_add, c.INPUT_MAX_DEPTH))
        np.append(part, to_add)
        unsplitted = np.concatenate([part, to_add])

    size = (np.shape(unsplitted))[0]
    return np.array(np.split(unsplitted, round(size / c.INPUT_MAX_WIDTH)))


# transforms xml files into ScoreParts
def get_score_parts(path = c.SCORES_PATH, is_single = False) -> list[ds.ScorePart]:
    score_parts = []

    if is_single:
        score_tree = ET.parse(path)
        root = score_tree.getroot()
        parts = root.findall("part")
        for part in parts:
            measures = part.findall("measure")
            converted_measures = map(lambda measure: dr.XMLConverter.transform_measure(measure), measures)
            score_parts.append(ds.ScorePart(path, list(converted_measures)))

        return score_parts

    scores = os.listdir(path)
    for score in scores:
        score_tree = ET.parse(path + '/' + score)
        root = score_tree.getroot()
        parts = root.findall("part")
        for part in parts:
            measures = part.findall("measure")
            converted_measures = map(lambda measure: dr.XMLConverter.transform_measure(measure), measures)
            score_parts.append(ds.ScorePart(score, list(converted_measures)))

    return score_parts
