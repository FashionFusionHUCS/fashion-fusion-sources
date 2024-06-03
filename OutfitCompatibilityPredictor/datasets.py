import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from CLIP import clip

class OutfitsDataset(Dataset):
    def __init__(self, image_dataframe, text_dataframe, transform=None):
        self.image_dataframe = image_dataframe
        self.text_dataframe = text_dataframe
        assert len(image_dataframe) == len(text_dataframe), "Number of outfits for images and descriptions do not match."
        self.transform = transform

    def __len__(self):
        return(len(self.image_dataframe))

    def __getitem__(self, idx):
        image_row = self.image_dataframe.iloc[idx]
        tops = Image.open(image_row['tops']).convert('RGB')
        bottoms = Image.open(image_row['bottoms']).convert('RGB')
        shoes = Image.open(image_row['shoes']).convert('RGB')
        outerwear = Image.open(image_row['outerwear']).convert('RGB')
        bags = Image.open(image_row['bags']).convert('RGB')
        
        if self.transform:
            tops = self.transform(tops)
            bottoms = self.transform(bottoms)
            shoes = self.transform(shoes)
            outerwear = self.transform(outerwear)
            bags = self.transform(bags)
        
        compatibility = torch.tensor(image_row['compatibility'], dtype=torch.float32)

        text_row = self.text_dataframe.iloc[idx]
        x1_tensor = clip.tokenize(["This is " + text_row['tops']], truncate=True).squeeze()
        x2_tensor = clip.tokenize(["This is " + text_row['bottoms']], truncate=True).squeeze()
        x3_tensor = clip.tokenize(["This is " + text_row['shoes']], truncate=True).squeeze()
        x4_tensor = clip.tokenize(["This is " + text_row['outerwear']], truncate=True).squeeze()
        x5_tensor = clip.tokenize(["This is " + text_row['bags']], truncate=True).squeeze()
        desc_tensors = torch.stack([x1_tensor, x2_tensor, x3_tensor, x4_tensor, x5_tensor], dim=0)

        return (tops, bottoms, shoes, outerwear, bags), desc_tensors, compatibility