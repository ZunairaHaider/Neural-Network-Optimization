# Publishing this case study as one GitHub repository

This folder is **one repository** that merges all five export archives. Upload its **contents** to a single GitHub repository (recommended name: `neural-network-optimization`). Do not upload the ZIP as a single file inside GitHub and do not create five repositories.

## Using the GitHub website

1. Create one empty public GitHub repository (do not initialize it with a different README).
2. Extract the downloaded ZIP on your computer.
3. Open the extracted `Neural-Network-Optimization` folder.
4. Upload the contents into the repository root. GitHub's browser upload can be inconvenient for thousands of files, so **GitHub Desktop or Git is recommended**.

## Using GitHub Desktop

1. In GitHub Desktop, select **File → Add local repository** and choose the extracted folder. If prompted, create a repository there.
2. Commit all intended files.
3. Select **Publish repository**; choose **Public**.

## Using a terminal

```bash
cd Neural-Network-Optimization
git init
git add .
git commit -m "Publish TUM-Schaeffler neural network optimization case study"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/neural-network-optimization.git
git push -u origin main
```

Create the empty remote repository first; replace `YOUR-USERNAME`. GitHub Desktop is easier if you are not comfortable with the terminal.

## Verification and notes

- Start with `README.md`, then `docs/REPRODUCIBILITY.md`.
- This edition preserves the supplied models, logs, numerical results, and report, including historical runs. Not all headline report results have a corresponding saved run; that limitation is documented.
- The bundled smoke test tests core components, not the full experimental pipeline.
- The complete working tree contains thousands of files. Commit/push from GitHub Desktop or the terminal instead of manually selecting files in the web interface.
