# Bachelor's Thesis Project

Welcome to my Bachelor's Thesis Project.  
You can find the thesis text in [`Abschlussarbeit_458692.pdf`](./Abschlussarbeit_458692.pdf).

---

## Branch Overview
- `main` — LaTeX source for the written thesis.  
- `sync`, `nonir`, `nonir-improved` — implemented pipelines (with/without NIR channel).  
- `coral` — Google Coral Dev Board Mini inference software (C++ with TF Lite API).  

See my thesis for further details on Coral setup.

---

## Pipelines Setup Instructions

1. **Download the dataset**  
   Get the [38-Cloud dataset](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images).

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt

3. **Adjust paths via config.json. Example:** 
```bash
{
  "baseDir": "C:/Users/andre/Documents/BA/dev/pipeline/Data",
  "csvTrainVal": "C:/Users/andre/Documents/BA/dev/pipeline/Data/training_patches_38-cloud_nonempty.csv",
  "csvTest": "C:/Users/andre/Documents/BA/dev/pipeline/Data/!myCSVs/fullTestDS.csv"
}
```

## Contact
For questions on implementation or reproducibility, feel free to reach out:
zharski@campus.tu-berlin.de

## 🧟‍♂️ Git Horror Stories Zone

This project’s commit history contains genuine artifacts of panic-merges, last-second backup tags, and at least one “whoops, closed the merge editor” event. All code and branches are healthy and working—just beware the haunted zig-zags you may find in the git graph.

**Future Me:**  
If you’re reading this, it all worked out.  
Enjoy the artifacts. May they remind you to always double-check your merge commits, and never fear a messy history.  
If you dare to clean it up, remember:  
`git tag backup-pre-cleanup` first!

*—The Ghost of Debugging Past 👻*