import torch
import random
import numpy as np
import warnings

RANDOM_SEED = 2025
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

warnings.filterwarnings("ignore", message="Workbook contains no default style")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=4, sci_mode=False)
print(f"Current Device: {device}")
