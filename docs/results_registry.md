# Historical results registry and data provenance

`results/index.csv` lists run labels, result-evidence classes, missing inputs, and scientific status. `results/archive_checksums.json` records SHA-256 and byte size of the two original `.7z` uploads so a separately imported LFS copy can be verified.

## Full archives **not** embedded in this starter ZIP

| Run | Original archive | Uploaded compressed size | Original archive contents |
|---|---|---:|---|
| Uniform | `synthetic_uniform_u8_vm20_run_55.7z` | ~84 MiB | 99 PKL, 99 WXF, 99 validation NPZ, diagnostics, QC CSV and video |
| Rankine | `synthetic_rankine_R80_vb3_cx255p5_cy255p5_run_11.7z` | ~181 MiB | 99 PKL, 99 WXF, 99 validation NPZ, diagnostics, QC CSV and video |

The full archives were user-provided but are **not included** in this smaller documentation ZIP. To meet the supervisor's “ALL historical and current results” requirement, add them to `results/archives/` as separate tracked Git LFS files and retain them unchanged. This project’s `.gitattributes` opts large `.7z`, TIFF, and MP4 files into LFS; consider `.npz` if you later store individual large arrays rather than 7z archives.

```bash
git lfs install
git add .gitattributes
# Copy original .7z archives into results/archives/ then:
git add results/archives/*.7z
git commit -m "Archive historical uniform and Rankine results via Git LFS"
```

The `results/examples` folders contain intentionally **incomplete illustrative subsets**: archived parameters, QC data, a single pair's validation NPZ, and diagnostic figures. They are convenient for browsing, but **do not replace the full run archives**. To support a paper, also archive ground truth, source TIFF sequence, original generator configuration, code commit and environment for each run.

## Provenance and immutability convention

Recommended per-run metadata: unique run label; original and current status; exact source commit; generator code commit; input and ground truth SHA-256; image dimensions; physical scale and frame period if relevant; seed; full parameter JSON; versions of Python/OpenCV/NumPy/SciPy; complete QC/output file list; evaluation script version; reviewer notes; reasons for invalidation/supersession. Do not edit old results in place; create new run IDs and record relations in the index.

## Existing source-repository archives

Additional LFS archives are retained unchanged in `Test_image_sequences/`, `Example_2WCF-SVD_outputs/` and `Artifact_mode_detection/Fourier_SVD_filtering_tests/` from the original `main` history. The registry CSV currently indexes **only** the two newly supplied 99-pair example archives, not the complete repository history. Complete this index before publication.
