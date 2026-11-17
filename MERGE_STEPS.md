# Step-by-step: merge modular and legacy 2WCF into your existing GitHub repository

This guide is tailored to the ZIP snapshot of `pienins/Optical-flow-svd-and-fourier-filtering` `main` uploaded on 17 Nov 2026. **The package `2wcf_merge_additions.zip` is an overlay of NEW files only.** It intentionally contains no original `Test_image_sequences/` or `Example_2WCF-SVD_outputs/` content and no historical `.7z` Git LFS pointers. Those original files remain untouched in the live repository.

> **Do not replace GitHub `main` using the downloaded main-branch ZIP.** ZIP downloads of Git-LFS repositories contain pointer text instead of actual large archives. Clone the real repository instead.

## A. Before starting (macOS Terminal)

Install Git and Git LFS if necessary. From the repository's real GitHub page, verify that you have push permission. Create a clean working directory, for example `~/Projects`.

```bash
cd ~/Projects
git clone https://github.com/pienins/Optical-flow-svd-and-fourier-filtering.git
cd Optical-flow-svd-and-fourier-filtering
git lfs install
git lfs pull
git status --short
```

If the repository already exists locally, do not clone it again; open its directory, finish/commit unrelated local changes, and run `git pull --ff-only` on `main` before branching. If Git LFS is unavailable, install it before working with the original archive files. GitHub login may require a credential manager or personal access token.

## B. Create an isolated branch

```bash
git switch main
git switch -c integration/modular-2wcf-and-docs
```

This branch prevents accidental changes to `main` until you have reviewed the pull request.

## C. Add the new modular code and docs without touching old results

Download `2wcf_merge_additions.zip` from the ChatGPT reply into your Downloads directory. It must be the *additions* package, not the historical source snapshot. From the repository's root:

```bash
unzip -n ~/Downloads/2wcf_merge_additions.zip -d .
git status --short
```

`-n` never overwrites an existing file. A clean clone of the uploaded `main` should not contain any of the new paths. If the live branch changed since the uploaded snapshot, inspect conflicts instead of forcing overwrite. **Do not use `unzip -o` blindly.**

The new files include `run.py`, `two_wcf/`, `config.json`, requirements, `tests/`, `docs/`, `benchmarks/`, small example datasets, and selected archived visualizations/metrics. The `config.json` is an exact copy of the uploaded refactor settings and points to a private `E:/` path. Use the portable `examples/mini_uniform_config.json` for initial verification.

## D. Move old v4.4 into `legacy/` (one explicit, reversible Git move)

The additions ZIP contains a `legacy/README.md` but **no duplicate v4.4 algorithm file**. This command moves your existing old script and retains Git's version history:

```bash
git mv 2WCF-SVD_v4_4.py legacy/2WCF-SVD_v4_4.py
```

Keep `Generation_testing_unfirom_flow/`, `Artifact_mode_detection/`, `Test_image_sequences/`, `Example_2WCF-SVD_outputs/`, `OFV_field_postprocessing_v2.nb`, and existing environment scripts in their original locations for this first merge. Moving dozens of historical paths in the same commit would make audit/recovery harder. Do not remove the old scripts or data.

## E. Verify the merged code

From the repository root, create a fresh virtual environment (separate from any Windows `fieldenv` setup script):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python benchmarks/generators/generate_mini_uniform.py
python run.py --config examples/mini_uniform_config.json
python benchmarks/evaluators/evaluate_mini_uniform.py
```

The mini example runs on 6 frames and outputs 5 adjacent-frame validations and a small EPE plot; it does **not** reproduce the historical 99-pair `u8_vm20` run. The generated local `output/` is ignored by Git. The runner refuses to overwrite an existing nonempty output directory by default, so on rerun use a new demo runID or delete only this disposable local `output/mini_uniform_run_1`.

**Important:** The original `setup_fieldenv.bat` removes an existing Conda environment named `fieldenv`. Avoid running it as a casual installation step.

## F. Inspect and commit your integration branch

```bash
git status --short
git diff --stat
git lfs ls-files
```

After checking the changed-file list and confirming no unexpected original large files are staged:

```bash
git add README.md MERGE_STEPS.md .gitignore config.json requirements.txt requirements-dev.txt pytest.ini run.py two_wcf tests benchmarks examples docs legacy data/examples results
# The git mv is staged automatically.
git diff --cached --stat
git commit -m "Integrate modular 2WCF and research documentation; preserve legacy"
git push -u origin integration/modular-2wcf-and-docs
```

Open the link GitHub prints to create a pull request into `main`. Review the `Files changed` tab, especially that old `.7z` files remain present and untouched, that only the old Python script was moved, and that the new README links work. Merge after review. The package does not change GitHub by itself.

## G. In a SEPARATE data commit: add the two later result archives via Git LFS

The two archives uploaded to ChatGPT are not bundled into this lightweight additions ZIP. Place your original files in the repository using Git LFS, preserving the names exactly:

```bash
git switch -c data/add-uniform-and-rankine-archives main
mkdir -p results/archives
cp ~/Downloads/synthetic_uniform_u8_vm20_run_55.7z results/archives/
cp ~/Downloads/synthetic_rankine_R80_vb3_cx255p5_cy255p5_run_11.7z results/archives/
# The existing .gitattributes already tracks all *.7z with Git LFS.
git add results/archives/*.7z
git lfs ls-files
git diff --cached --stat
git commit -m "Archive uniform and Rankine experiment outputs using Git LFS"
git push -u origin data/add-uniform-and-rankine-archives
```

Use `sha256sum` on Linux, `shasum -a 256` on macOS, or `Get-FileHash` on Windows to compare the local archives with `results/archive_checksums.json` before committing. Uploaded archive sizes are approximately 84 MiB and 181 MiB. Git LFS usage counts toward your storage/bandwidth quota. If these files are *already* present in the live repo under different paths, don't duplicate them without a clear reason.

## H. What's still missing for scientific reproducibility?

See [`docs/missing_materials.md`](docs/missing_materials.md). In particular: independent truth maps and original input TIFFs for both late archived runs, Rankine generator, historical `frame2_polynomial` solver revision, run provenance, complete historic sweep/PSO outputs, author-approved license and citation metadata, and controlled benchmark comparisons.

## Windows alternative

Use Git Bash, or equivalent PowerShell commands. In Git Bash, the above `git` and `unzip` commands are nearly identical; use `/c/Users/<account>/Downloads/` instead of `~/Downloads/` if necessary. In PowerShell, extract the overlay with `Expand-Archive` into the checked-out repository after inspecting contents; do not overwrite the original repo blindly.
