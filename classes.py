"""Class definitions: real + 9 Synthbuster generators, and the family hierarchy."""
import re

CLASSES = ["real", "dalle2", "dalle3", "firefly", "glide",
           "midjourney", "sd13", "sd14", "sd2", "sdxl"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

FAMILIES = ["real", "stable_diffusion", "dalle", "other"]
NUM_FAMILIES = len(FAMILIES)
_FAMILY_OF = {"real": 0,
              "sd13": 1, "sd14": 1, "sd2": 1, "sdxl": 1,
              "dalle2": 2, "dalle3": 2,
              "firefly": 3, "glide": 3, "midjourney": 3}
CLASS_FAMILY = [_FAMILY_OF[c] for c in CLASSES]  # class idx -> family idx

# normalised folder name (lowercase, alphanumerics only) -> canonical class
_ALIASES = {
    "real": "real", "raise": "real", "raise1k": "real", "original": "real", "authentic": "real",
    "dalle2": "dalle2", "dalle3": "dalle3", "firefly": "firefly", "glide": "glide",
    "midjourney": "midjourney", "midjourneyv5": "midjourney",
    "stablediffusion13": "sd13", "sd13": "sd13",
    "stablediffusion14": "sd14", "sd14": "sd14",
    "stablediffusion2": "sd2", "sd2": "sd2",
    "stablediffusionxl": "sdxl", "sdxl": "sdxl",
}


def canonical_class(folder_name):
    key = re.sub(r"[^a-z0-9]", "", folder_name.lower())
    return _ALIASES.get(key)