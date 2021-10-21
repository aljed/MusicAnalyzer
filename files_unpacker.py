import zipfile as zf
import constants as c
import os



def unzip_scores():
    mxl_files = os.listdir(c.MXL_FILES_PATH)
    for f in mxl_files:
        zip_ref = zf.ZipFile(c.MXL_FILES_PATH + '\\' + f)
        zip = zip_ref.namelist()
        xml_file_name = filter(lambda name: name.endswith('xml') and not name.startswith('META-INF'), zip).__next__()
        zip_ref.extract(xml_file_name, c.SCORES_PATH)
        os.rename(c.SCORES_PATH + '/' + xml_file_name, c.SCORES_PATH + '/' + f.replace('mxl', 'xml'))


def list_score_names():
    xml_files = os.listdir(c.SCORES_PATH)
    for name in xml_files:
        print(name)

list_score_names()
