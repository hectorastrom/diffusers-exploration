# @Time    : 2025-11-24 15:51
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : COD_dataset.py

import os
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, Features, Image as ImageFeature, ClassLabel, Value
from reward import CLIPReward

##################################
# Dataset Reading
##################################
# [SCRIPT_DIR]/datasets/COD10K
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATASET_ROOT = os.path.join(SCRIPT_DIR, "datasets", "COD10K")

SPLITS = ["Train", "Test"]
# SubClass is located at index 5 in the filename split.
LABEL_INDEX = 5

def get_label_from_filename(filename: str) -> str:
    """Extracts the SubClass label from the COD10K filename."""
    parts = filename.split('-')
    
    if len(parts) <= LABEL_INDEX:
         # no longer returning superclass label, because they don't make sense
         # these None will be filtered out later
        return None
    
        # Get next best label at idx 3 - the superclass label
        if len(parts) <= 3:
            raise Exception(f"Malformed file name (too short): {filename}")
        else: 
            return parts[3]
    
    # Extract the SubClass name (e.g., 'Snake')
    return parts[LABEL_INDEX]

def image_label_generator(split_name: str):
    # Generators are memory efficient, lazy, and expected by datasets library
    """
    Generator function to yield (image, label_name) pairs.
    
    Note that images are stored as image_paths, and decoded by datasets.
    """
    # 'Train/Images' and 'Test/Images' structure
    split_path = os.path.join(DATASET_ROOT, split_name, "Image")
    
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Directory not found: {split_path}")

    for filename in sorted(os.listdir(split_path)):    
        if filename.endswith(".jpg"): # only jpg - i checked
            file_path = os.path.join(split_path, filename)
            label_name = get_label_from_filename(filename)
            
            yield {
                "image": file_path, # same as image_path for now, but will be decoded to img
                "image_path": file_path, 
                "label_name": label_name
            }
        else:
            print(f"Not an image (harmless, skipped): {filename}")

def load_cod10k_lazy() -> DatasetDict:
    """
    Loads COD10K dataset with subclass as label. Lazy loading only loads
    filepaths, which are automatically decoded to images through datasets'
    ImageFeature() schema.
    
    Cols: 'image', 'label', 'image_path'
    """
    assert os.path.isdir(DATASET_ROOT), "ERORR: Dataset folder is missing."
    dataset_dict = {}
    
    # schema is info datasets need to load image on demand from filepath
    features_schema = Features({
        "image" : ImageFeature(),
        "label_name": Value('string'),
        "image_path": Value('string')
    })
    
    for split in SPLITS:
        raw_dataset = Dataset.from_generator(
            image_label_generator,
            features=features_schema,
            gen_kwargs={"split_name": split} # this are kwargs it passes to image_label_generator
        )
        dataset_dict[split.lower()] = raw_dataset
        
    raw_datasets = DatasetDict(dataset_dict)
    
    # filter out superclasses
    # takes train dataset from 6000 -> 4190 images
    raw_datasets = raw_datasets.filter(
        lambda x: x['label_name'] is not None,
        num_proc=os.cpu_count() // 2
    )
    
    # convert str class names to ClassLabel features
    all_labels = set()
    for split in raw_datasets:
        all_labels.update(raw_datasets[split]["label_name"])
        
    label_list = sorted(list(all_labels))
    # mapping of all possible labels to ints (categorical)
    class_feature = ClassLabel(names=label_list) 
    
    def encode_labels(sample):
        sample['label'] = class_feature.str2int(sample['label_name'])
        return sample
    
    print("Encoding labels...")
    raw_datasets = raw_datasets.map(encode_labels)
    
    # imgs remain untouched, and are decoded from filepaths JIT
    final_dataset = raw_datasets.map(
        encode_labels, 
        remove_columns=['label_name']
    )
    
    final_dataset = final_dataset.cast_column("label", class_feature)
    
    return final_dataset

##################################
# Transforming to torch dataset
##################################

def build_transform(image_size):
    tensor_transform = transforms.Compose([
        # add data augmentation methods
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        # 512x512 works best for SD1.5 - but CLIP works with 224x224 images
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    def pytorch_transform_fn(examples):
        # examples is a dict of lists: {'image': [PIL, ...], 'label': [0, ...]}
        examples['pixel_values'] = [tensor_transform(img.convert("RGB")) for img in examples['image']]
        del examples['image'] # don't need for loader
        return examples

    return pytorch_transform_fn

def build_COD_torch_dataset(split_name = 'train', image_size=512):
    dataset = load_cod10k_lazy()[split_name]
    dataset.set_transform(build_transform(image_size=image_size))
    
    label_feature = dataset.features['label']
    # Adding some metadata that's helpful
    dataset.all_classes = label_feature.names 
    dataset.label2str = label_feature.int2str
    
    return dataset
    
if __name__ == "__main__":
    dataset = build_COD_torch_dataset('train')
    loader = DataLoader(dataset, batch_size=4, num_workers=1, shuffle=True)
    
    reward_fn = CLIPReward(dataset.all_classes, device=0)
    
    for item in loader: # dict{'label', 'pixel_values'}
        img_tensors = item['pixel_values']
        label_ints = item['label']
        image_paths = item['image_path']
        print("Image tensor:", img_tensors.shape)
        print(img_tensors.min(), img_tensors.max()) # images are [0, 1]
        print("Labels:", label_ints)
        label_str = dataset.label2str(label_ints)
        print("Str labels:", label_str)
        
        print(f"Image paths: {image_paths}")
        
        meta = [{'label' : int(i)} for i in label_ints]
        
        rewards, _ = reward_fn(img_tensors, prompts=[''] * len(img_tensors), metadata=meta)
        print(f"Batch reward {rewards}")
        print(f"Mean reward: {rewards.mean().item()}")
        break
