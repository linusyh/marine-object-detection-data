from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pascal import PascalVOC
import xmltodict
from PIL import Image
import sys

def check_fix_xml(xml_path: Path, try_pascalvoc = True):
    with open(xml_path) as f:
        raw_xml = f.read()
        if len(raw_xml.strip()) == 0:
            print(f"emoty xml file {xml_path}")
            return True

        dict_xml = xmltodict.parse(raw_xml)
        if dict_xml['annotation']['size']['width'] == "None" or dict_xml['annotation']['size']['height'] == "None":
            stored_path = Path(dict_xml['annotation']['path'])
            img_fname = stored_path.name
            img_path = xml_path.parent / img_fname
            img = Image.open(img_path)
            width, height = img.size
            depth = len(img.getbands())
            dict_xml['annotation']['size']['width'] = width
            dict_xml['annotation']['size']['height'] = height
            dict_xml['annotation']['size']['depth'] = depth
            print(f"Fixing {xml_path.name} {width} {height} {depth}")
            xmltodict.unparse(dict_xml, open(xml_path, "w"))

            if try_pascalvoc:
                _ = PascalVOC.from_xml(xml_path)

        else:
            return False

if __name__ == "__main__":
    dir = Path(sys.argv[-1])
    executor = ThreadPoolExecutor(max_workers=64)
    count = 0
    total = 0
    for b in executor.map(check_fix_xml, dir.glob("**/*.xml")):
        if b:
            count += 1
    print(f"total {count} corrected")