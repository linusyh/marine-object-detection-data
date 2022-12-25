from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path

import cv2
from pascal import PascalVOC


def cropout(xml_path, output_dir):
    xml_path = Path(xml_path)
    try:
        ann = PascalVOC.from_xml(xml_path)
    except Exception as e:
        print(f"{xml_path}: PascalVOC error {e}")
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    concept = str(xml_path.parent.parts[-1]).replace('_', " ") # DEPENDS ON FILE LAYOUT
    image_path = xml_path.parent / ann.filename

    img = cv2.imread(image_path.as_posix())
    index = 0
    for obj in ann.objects:
        if obj.name == concept:
            output_path = output_dir / f"{(image_path.stem)}-{index}{image_path.suffix}"
            index += 1

            if output_path.exists():
                return 1

            try:
                cropped = img[obj.bndbox.ymin: obj.bndbox.ymax, obj.bndbox.xmin: obj.bndbox.xmax]
            except Exception as e:
                print(e)
                return 0
            
            h, w, c = cropped.shape
            if h == 0 or w == 0 or c==0:
                print(f"Empty boundingbox")
                continue
            cv2.imwrite(output_path.as_posix(), cropped)
    
    return index


def cropoutDir(xml_dir, output_dir):
    for xml_file in glob(f"{xml_dir}/*.xml"):
        cropout(xml_file, output_dir)


def cropOutTree(input_dir, output_dir_root):
    input_dir = Path(input_dir)
    output_dir_root = Path(output_dir_root)

    def _crop(xml_file):
        output_dir = output_dir_root.joinpath(*xml_file.parent.parts[len(input_dir.parts):])
        return cropout(xml_file, output_dir)

    
    total = 0
    with ThreadPoolExecutor(max_workers=100) as executor:
        for x in executor.map(_crop, input_dir.glob("**/*.xml")):
            total += x
            if (total - x)//500 < total//500:
                print(f"{input_dir}: {total} cropped")



cropOutTree("./downloads", "./cropped/")