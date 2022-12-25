from pathlib import Path
import sys

import pandas as pd
from pascal import PascalVOC
from tqdm import tqdm

def create_df_from_xmls(xml_dir: Path):
    # df = pd.DataFrame(columns=[
    #     "uuid", "index", "img_w", "img_h", "bbox_x", "bbox_y", "bbox_width", "bbox_height"
    # ])

    rows = {}
    rows['uuid'] = []
    rows['super_concept'] = []
    rows['specific_concept'] = [] 
    rows['index'] = [] 
    rows['img_w'] = [] 
    rows['img_h'] = [] 
    rows['ann_x'] = [] 
    rows['ann_y'] = [] 
    rows['ann_w'] = [] 
    rows['ann_h'] = [] 

    for f in tqdm(xml_dir.glob("**/*.xml")):
        try:
            ann = PascalVOC.from_xml(f)
        except Exception as e:
            print(f"{f}: PascalVOC error {e}")
            continue
        
        super_concept = str(f.parent.parts[1])
        specific_concept = str(f.parent.parts[-1]).replace('_', " ") # DEPENDS ON FILE LAYOUT
        uuid = f.stem
        img_w = ann.size.width
        img_h = ann.size.height        
        index = 0
        for obj in ann.objects:
            if obj.name == specific_concept:
                rows['uuid'].append(uuid)
                rows['super_concept'].append(super_concept)
                rows['specific_concept'].append(specific_concept)
                rows['index'].append(index)
                rows['img_w'].append(img_w)
                rows['img_h'].append(img_h)
                rows['ann_x'].append(obj.bndbox.xmin)
                rows['ann_y'].append(obj.bndbox.ymin)
                rows['ann_w'].append(obj.bndbox.xmax - obj.bndbox.xmin)
                rows['ann_h'].append(obj.bndbox.ymax - obj.bndbox.ymin)
                index += 1
    
    return pd.DataFrame.from_dict(rows)

    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Syntax: {sys.argv[0]} <xml_dir> <csv file>")
        sys.exit(1)
    
    df = create_df_from_xmls(Path(sys.argv[1]))
    df.to_csv(Path(sys.argv[2]))