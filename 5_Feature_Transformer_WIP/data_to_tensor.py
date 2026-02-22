import torch
from torch.utils.data import Dataset

# To use the FT transformers and a lot of other transformers we must use tensors. Here we myst convert dataset to tensor format
#note to self: data cannot be NAN, fill or drop, labekl encoding where categorical columns need to be 0-1
class ACSDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, target_col):
        # Converting  columns to Tensors for init
        self.x_cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.x_cont = torch.tensor(df[cont_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # alloqing dataloder to grab one row at a time
        return self.x_cat[idx], self.x_cont[idx], self.y[idx]