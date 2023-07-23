from kaggle_llm.core import WORK_DIRS_PATH, ROOT_PATH
from tqdm import tqdm
import zipfile
import yaml


take_dirs = [
    "configs",
    "src",
    "tools",
    "source.bash",
]
with open(ROOT_PATH / "configs/submission.yaml", "rb") as f:
    configs = yaml.load(f, yaml.FullLoader)


def main():
    i = 0
    with zipfile.ZipFile(ROOT_PATH / "IGNORE_ME_archive.zip", "w") as z:
        for m in tqdm(configs["models"]):
            model_path = WORK_DIRS_PATH / m
            found_paths = [p for p in model_path.glob("*") if p.name != "optimizer.pt"]
            i += len(found_paths)
            print(f"found {len(found_paths)} items under {m}")
            for p in found_paths:
                z.write(p, p.relative_to(ROOT_PATH), zipfile.ZIP_DEFLATED)
        for d in tqdm(take_dirs):
            full_path = ROOT_PATH / d
            if full_path.is_file():
                found_paths = [full_path]
            else:
                found_paths = list(full_path.rglob("*"))
            i += len(found_paths)
            print(f"found {len(found_paths)} items under {d}")
            for p in found_paths:
                z.write(p, p.relative_to(ROOT_PATH), zipfile.ZIP_DEFLATED)
    print(f"done: added {i} files")


if __name__ == "__main__":
    main()
