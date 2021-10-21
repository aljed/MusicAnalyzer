import xml.etree.ElementTree as ET
import os
import data_structure as ds
import data_retriever as dr
import constants as c

scores = os.listdir(c.SCORES_PATH)

for score in scores:
    if score != 'Chopin_-_Nocturne_Op_9_No_2_E_Flat_Major (1).xml':
        continue
    score_tree = ET.parse(c.SCORES_PATH + '/' + score)
    root = score_tree.getroot()
    parts = root.findall("part")
    for part in parts:
        measures = part.findall("measure")
        converted_measures = map(lambda measure: dr.XMLConverter.transform_measure(measure).moments, measures)
        print(((list(converted_measures))[1][2]).encode())
