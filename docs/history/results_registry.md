# Historical results registry and data provenance

*Historical record; reviewed 24 November 2026.*

`results/index.csv` lists run labels, result-evidence classes, missing inputs and scientific status. `results/archive_checksums.json` records the SHA-256 and byte size of the two original 99-pair `.7z` output archives, so that copies can be verified byte for byte.

## The two 99-pair archives

| Run | Archive | Size | Contents |
|---|---|---:|---|
| Uniform | `synthetic_uniform_u8_vm20_run_55.7z` | 87 765 348 bytes (~84 MiB) | 99 PKL, 99 WXF, 99 validation NPZ, diagnostics, QC CSV and video |
| Rankine | `synthetic_rankine_R80_vb3_cx255p5_cy255p5_run_11.7z` | 189 621 444 bytes (~181 MiB) | 99 PKL, 99 WXF, 99 validation NPZ, diagnostics, QC CSV and video |

**These archives are not yet in the repository.** `results/archives/` contains only a README. The project aims to keep all historical and current results, so they should be added there unchanged. The existing `*.7z` rule in `.gitattributes` already sends them to Git LFS; no other file types are tracked by LFS. Check each file against `results/archive_checksums.json` first (`sha256sum` on Linux, `shasum -a 256` on macOS, `Get-FileHash` in PowerShell):

```bash
git lfs install
# copy the two .7z files into results/archives/, verify checksums, then:
git add results/archives/*.7z
git lfs ls-files            # both archives must be listed here
git commit -m "Archive historical uniform and Rankine results via Git LFS"
```

Together they add about 265 MiB to Git LFS storage, and each full clone with `git lfs pull` downloads them (see the LFS note in [status and limitations](../status_and_limitations.md) §5).

Neither archive contains its input TIFF sequence or `true_vector_values.npz`, so they are output archives, not reproduction bundles.

## Example excerpts

`results/examples/` holds deliberately **incomplete subsets** of the two runs: archived parameters, QC data, the first pair's validation NPZ and diagnostic figures ([examples](examples.md)). They are convenient for browsing but **do not replace the full archives**. To support a paper, also archive the ground truth, input TIFF sequence, generator configuration, code commit and environment of each run.

## Provenance and immutability convention

Recommended per-run metadata: unique run label; original and current status; exact source commit; generator code commit; input and ground-truth SHA-256; image dimensions; physical scale and frame period if relevant; seed; full parameter JSON; versions of Python/OpenCV/NumPy/SciPy; complete QC/output file list; evaluation script version; reviewer notes; reasons for invalidation or supersession. Do not edit old results in place; create new run IDs and record their relations in the index.

## Other archives in the repository

`experimental_data/` (moved from `Test_image_sequences/` and `Example_2WCF-SVD_outputs/`) and `Artifact_mode_detection/Fourier_SVD_filtering_tests/` hold 11 further LFS archives (about 310 MB in total), unchanged in content since the original `main` history ([`experimental_data/README.md`](../../experimental_data/README.md)). `results/index.csv` indexes **only** the two 99-pair runs, not these; complete the index before publication.
