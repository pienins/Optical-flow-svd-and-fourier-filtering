# Setup and running

Written for Windows with Anaconda. Other systems: see the end of this page.

## 1. Get the repository

In **GitHub Desktop**: File → Clone repository → `pienins/Optical-flow-svd-and-fourier-filtering`. GitHub Desktop also downloads the large `.7z` archives (Git LFS).

Do not use GitHub's "Download ZIP": it contains small placeholder files instead of the `.7z` archives. Unpack an archive with 7-Zip when you need its images.

## 2. Install the environment (once)

Open **Anaconda Prompt**, go to the repository folder and run the setup script:

```text
cd C:\path\to\Optical-flow-svd-and-fourier-filtering
flowenv_setup.bat
```

It creates the Conda environment `flowenv` with the tested package versions, Spyder and pytest. It also stores `PYTHONNOUSERSITE=1` in the environment, so Python ignores packages installed for your Windows user outside the environment.

## 3. Start working (each session)

In Anaconda Prompt:

```text
conda activate flowenv
spyder
```

## 4. Run the pipeline

1. Open `config.json` and set:
   - `paths.input_images`: the folder containing the TIFF frames;
   - `paths.output_root`: where results should go;
   - `run.runID`: a new number for each run.
2. Open `run.py` in Spyder and press **Run**. It uses the `config.json` next to `run.py`.

To use a different config file: in Spyder, Run → Configuration per file → Command line options: `--config C:\path\to\other_config.json`. From Anaconda Prompt: `python run.py --config C:\path\to\other_config.json`.

## Rules to know

- **Results** go to `<output_root>\<name>_run_<runID>\`, where `<name>` is the input folder's name (or its parent folder's name if the input folder is called `frames`). See [Output formats](output_formats.md) for what is inside.
- **Use a new `runID` for every run.** An existing, non-empty run folder stops the program. `outputs.OverwriteExistingRun = true` **deletes** that run folder first; use it only for disposable results.
- **Frame names:** frames are sorted by the **first** number in each filename. `frame_0001.tif`, `frame_0002.tif` is fine. `exp2_frame_0001.tif` is not: every frame sorts as "2", and the order is wrong without any error.
- **Frames** must be 2-D grayscale `.tif`/`.tiff` images of equal size, at least two, at equal time intervals.
- **Units:** results are displacements in **pixels per frame pair**, with positive `u` to the right and positive `v` downward. Physical velocity needs the pixel size and frame interval.
- **`outputs.WriteWXF = true`** (Mathematica files) needs `wolframclient`, which the setup script installs.
- `run.MaxFramePairs = 5` processes pairs 0→1 … 4→5; `null` processes all pairs.
- The whole image sequence is loaded into memory, also with `run.UseParallelFlow = true`.

## Checking the installation (optional)

In Anaconda Prompt, with `flowenv` active, in the repository folder:

```text
python -m pytest -q
```

It should end with `11 passed`. The tests are quick checks on small synthetic images, including one generate → pipeline → evaluate benchmark run. They are not a full numerical validation.

## Tested environment

The 11 tests passed (24 Nov 2026) with these versions, which are also those installed on the main Windows workstation by `flowenv_setup.bat`:

| Package | Version |
|---|---|
| Python | 3.10.20 |
| NumPy | 1.26.4 |
| OpenCV (`opencv-python-headless`) | 4.5.5 |
| SciPy | 1.15.3 |
| pandas | 2.3.3 |
| Matplotlib | 3.10.8 |
| tifffile | 2025.5.10 |
| imageio | 2.37.3 |
| tqdm | 4.67.3 |

The tests were run on Linux with these versions, not on the Windows workstation itself. They also passed with Python 3.13, NumPy 2.5 and OpenCV 5.0.

## Troubleshooting

| Symptom | Check |
|---|---|
| `EnvironmentNameNotFound: flowenv` | Run `flowenv_setup.bat` first (step 2), or list environments with `conda env list` |
| No input images found | `paths.input_images` must be the folder that contains the TIFFs, not its parent |
| Frames processed in the wrong order | Filenames: the first number in the name must be the frame number (see above) |
| Unknown or missing JSON key | Every group and parameter of the repository's `config.json` must be present; start from a copy of it |
| Output folder is not empty | Use a new `run.runID` |
| `wolframclient` import error | Set `outputs.WriteWXF = false`, or run `python -m pip install wolframclient` |
| `No module named pytest` | Run `python -m pip install pytest` in `flowenv` |
| Missing plots or video | Check the switches in `outputs` and `diagnostics` |
| Out of memory | Disable SVD, use fewer or smaller frames, or lower `MaxFlowWorkers` |
| Reproducing the archived Rankine run | Not possible with this code: it used a `frame2_polynomial` mode that the current solver does not have |

## Other systems (macOS, Linux)

In any Python 3.10 environment, from the repository folder:

```text
python -m pip install -r requirements-dev.txt
python -m pytest -q
python run.py --config /path/to/config.json
```

`requirements.txt` does not pin versions; use the versions above when results must match. Clone with Git LFS (`git lfs install`, then `git lfs pull`) to get the `.7z` archives.
