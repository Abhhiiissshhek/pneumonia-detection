"""Project-wide constants and default paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Defaults (all overridable from the CLI)
DATA_DIR = PROJECT_ROOT / "data" / "chest_xray"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_CHECKPOINT = MODELS_DIR / "pneumonia_resnet18.pt"

# Older versions of this repo saved a raw state_dict here; still supported.
LEGACY_CHECKPOINT = PROJECT_ROOT / "model" / "pneumonia_model.pth"

# Where the app downloads weights from if none exist locally.
# Upload the trained checkpoint as an asset of a GitHub Release with this name.
DEFAULT_MODEL_URL = (
    "https://github.com/Abhhiiissshhek/pneumonia-detection/releases/latest/download/pneumonia_resnet18.pt"
)

# ImageFolder sorts class folders alphabetically: NORMAL=0, PNEUMONIA=1
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
DISPLAY_NAMES = {"NORMAL": "Normal", "PNEUMONIA": "Pneumonia"}
POSITIVE_CLASS = 1

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEED = 42
