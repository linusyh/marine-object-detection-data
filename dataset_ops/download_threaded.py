from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import TimeoutError as cfTimeoutEror
from dataclasses import dataclass
from fathomnet.api import boundingboxes, images, taxa
from fathomnet.models import Taxa
from fathomnet.models import *
from pascal_voc_writer import Writer
import requests
import shutil

from tqdm import tqdm
from pathlib import Path
import treelib
from treelib import Tree
from typing import *
import json, pickle, sys, urllib3


def buildTree(concept: Taxa, provider='fathomnet', trimEmpty = True):
    def buildSubTree(_taxa: Taxa, tree: Tree, parent: str):
        direct_children_image_count = boundingboxes.count_by_concept(_taxa.name).count
        children_taxas = taxa.find_children(provider, _taxa.name)
        id_prefix = f"{_taxa.rank}_{_taxa.name}"
        
        total = direct_children_image_count
        if direct_children_image_count > 0:
            tree.create_node('direct', f"{id_prefix}_direct", parent, direct_children_image_count)

        for t in children_taxas:
            child_id = f"{parent}__{t.name.replace(' ', '_')}"
        
            try:
                tree.create_node(f"{t.rank} - {t.name}", child_id, parent, -1)
            except treelib.exceptions.DuplicatedNodeIdError:
                print(f"skip duplicate: {child_id}")
            
            child_total =  buildSubTree(t, tree, child_id)
            if child_total == 0 and trimEmpty:
                tree.remove_node(child_id)
                continue
        
            tree.get_node(child_id).data = child_total
            total += child_total
        
        return total

    tree = Tree()
    tree.create_node(f"{concept.rank} - {concept.name}", 'root', data=-1)
    total = buildSubTree(concept, tree, 'root')
    tree.get_node('root').data = total
    return tree


def buildTreeBetter(_taxa: Taxa, provider='fathomnet'):
    searchSpace = []
    leaves = set()
    
    def exploreSearchSpace(t: Taxa, parent):
        searchSpace.append((t, parent))
        new_parent = f"{parent}__{t.name.replace(' ', '_')}"
        children = taxa.find_children(provider, t.name)

        if len(children) == 0:
            leaves.add(f"{t.rank}-{t.name}")


        for c in children:
            exploreSearchSpace(c, new_parent)

   
    exploreSearchSpace(_taxa, "0")
    print(f"Search space size: {len(searchSpace)} / {len(leaves)} leaves")

    tree = Tree()
    tree.create_node("ROOT", "0", data = 0) 

    for t, parent in tqdm(searchSpace):
        children_image_count = boundingboxes.count_by_concept(t.name).count

        if children_image_count == 0 and (f"{t.rank}-{t.name}" in leaves):
            continue

        try:
            tree.create_node(f"{t.rank} - {t.name}", f"{parent}__{t.name.replace(' ', '_')}", parent, children_image_count)
        except treelib.exceptions.DuplicatedNodeIdError:
            print(f"skip duplicate: {parent}__{t.name}")
    
    cleanupTree(tree, "0")

    return tree


class MyTree(Tree):
    def show(self, labelFunc, printFunc, nid=None, level=0, filter=None, key=None, reverse=False, line_type='ascii-ex'):
        for pre, node in self._Tree__get(nid, level, filter, key, reverse, line_type):
            label = labelFunc(node)
            printFunc('{0}{1}'.format(pre, label))

    def showToFile(self, fileName,  labelFunc, **kwargs):
        with open(fileName, "w") as f:
            self.show(labelFunc, lambda l: f.write(l + '\n'), **kwargs)


@dataclass
class Count:
    count: int
    accumulated_count: int


def cleanupTree(tree: Tree, nid):
    total = tree.get_node(nid).data.count
    for cid in [c.identifier for c in tree.children(nid)]:
        total += cleanupTree(tree, cid)
    
    children = [c.identifier for c in tree.children(nid)]
    if len(children) == 0 and not total:
        tree.remove_node(nid)
        return 0
    else:
        tree.get_node(nid).data.accumulated_count = total
        return total


def build_phylogeny(conceptName: str, tree: Tree = None, rootID = 0, rootTag = 'ROOT'):
    http = urllib3.PoolManager()
    req = http.request('GET', f"https://fathomnet.org/dsg/phylogeny/down/{conceptName}")
    if req.status >= 300:
        print("Failed to connect to fathomnet phylogeny")
        return None
    
    dsg_data = json.loads(req.data)

    if not tree:
        tree = MyTree()
        tree.create_node(rootTag, rootID, data=Count(0, 0))
    
    def _buildTree(json_data, parent, tree: Tree):
        rank = json_data.get('rank') or 'unknown'
        
        tag = f"({rank}.){json_data['name']}"
        id = f"{parent}_{json_data['name'].replace(' ', '_')}"
        tree.create_node(tag, id, parent, Count(0, 0))

        if "children" in json_data.keys():
            for c in json_data["children"]:
                _buildTree(c, id, tree)

    _buildTree(dsg_data, rootID, tree)

    return tree


def add_counts_to_tree(phylenogy: Tree):
    progress = tqdm(enumerate(list(phylenogy.all_nodes())))
    for _, n in progress:
        if n.identifier == 0:
            continue

        concept = n.tag.split(')')[-1].strip()
        if len(concept) == 0:
            continue
    
        progress.set_postfix_str(concept)

        count = boundingboxes.count_by_concept(concept).count
        phylenogy.get_node(n.identifier).data.count = count


def build_concepts_tree(concepts: List[str], rootID: str, rootTag: Any, includeImgCount=False):
    if not rootTag:
        return None
    
    tree = MyTree()
    for c in concepts:
        tree = build_phylogeny(c, tree, rootID, rootTag)
    
    if tree:
        if includeImgCount:
            add_counts_to_tree(tree)
            cleanupTree(tree, rootID)
    else:
        print(f"Failed to build tree {rootTag} with concepts {concepts}")
    return tree


def download_images_data(concept):
    return images.find_by_concept(concept)


def write_annotation(image_data: AImageDTO, image_path: Path, output_dir: Path):
    writer = Writer(image_path, 
                    image_data.width,
                    image_data.height,
                    database='FathomNet')
    
    for box in image_data.boundingBoxes:
        concept = box.concept
        writer.addObject(concept,
                         box.x,
                         box.y,
                         box.x + box.width,
                         box.y + box.height)
        
    xml_path = output_dir / f"{image_data.uuid}.xml"
    writer.save(xml_path)


def download_and_annotate(image_data: AImageDTO, output_dir: Path, cached_dir: Path = None):
    url = image_data.url
    ext = Path(url).suffix
    uuid = image_data.uuid
    image_filename = f"{uuid}{ext}"
    metadata_filename = f"{uuid}.xml"
    image_path = output_dir / image_filename
    metadata_path = output_dir / metadata_filename
    
    if image_path.exists():
        return 0

    if cached_dir:
        cached_images = cached_dir.glob(f"**/{image_filename}")
        for f in cached_images:
            metadata_path_cached = f.parent / metadata_filename
            if metadata_path_cached.exists():
                shutil.copy(str(f), str(image_path))
                shutil.copy(str(metadata_path_cached), str(metadata_path))
                return 0
        
    resp = requests.get(url)
    
    if resp.status_code == 200:
        with open(image_path, "wb") as f:
            f.write(resp.content)
        write_annotation(image_data, image_path, output_dir)
    else:
        print(f"Error fetching {url}.")
    
    return 0;
    

def write_tree_metadata(tree: MyTree, name: str, output: Path):
    tree_file = output / f"{name}.tree"
    pickle_file = output / f"{name}.pickle"
    tree.showToFile(tree_file, lambda n: f"{n.tag} ({n.data.count})")
    with open(pickle_file, "wb") as f:
        pickle.dump(tree, f)


def repeat(x):
    while True:
        yield x


def download_tree(tree: MyTree, rootID, root_dir: Path, cached_dir: Path=None):
    def _download_node(node, download_dir):
        concept = node.tag.split(')')[-1]
        if len(concept)==0:
            return
        
        new_path = download_dir / concept.replace(' ', '_')        
        new_path.mkdir(parents=True, exist_ok=True)
        images_data = download_images_data(concept)
        
        print(f"{concept}: {new_path} ({len(images_data)})")
        
        executor = ThreadPoolExecutor(max_workers=40)
        try:
            [x for x in executor.map(download_and_annotate, images_data, repeat(new_path), repeat(cached_dir), timeout=30+len(images_data))]
        except cfTimeoutEror as e:
            print(f"Timeout: {e}")
        
        for c in tree.children(node.identifier):
            _download_node(c, new_path)

    for c in tree.children(rootID):
        _download_node(c, root_dir)


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        cached_dir = Path(sys.argv[3])
    output_dir = Path(output_dir)
    metadata_dir = output_dir / ".metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as f:
        class_concept_lkup = json.load(f)

    print(class_concept_lkup)
    
    for clsname, concepts in class_concept_lkup.items():
        print(clsname)
        class_dir = output_dir / clsname.replace(' ', '_')
        tree = build_concepts_tree(concepts, 0, clsname, True)
        tree.show(lambda n: f"{n.tag} ({n.data.count})", print)
        write_tree_metadata(tree, clsname.replace(' ', '_'), metadata_dir)
        download_tree(tree, 0, class_dir, cached_dir=cached_dir)

