# Third-party code and assets

The method-specific release code is licensed under Apache-2.0. Included upstream
files keep their existing notices; this release does not replace upstream licenses
or model/dataset provider terms.

| Component | Location | Upstream license text |
|---|---|---|
| Physical Intelligence OpenPI | `openpi/` | `openpi/LICENSE`, `licenses/openpi.txt` (Apache-2.0) |
| Lifelong Robot Learning LIBERO | `openpi/third_party/libero/` | its `LICENSE`, `licenses/LIBERO.txt` (MIT) |
| InternRobotics VLAC | `experiments/vlac/VLAC/` | its `LICENSE`, `licenses/VLAC.txt` (upstream MIT notice) |

The upstream VLAC package metadata and its LICENSE text may use different license
labels; both original files are retained unchanged. Consult the upstream project
for its authoritative terms rather than treating this package as a relicensing.

`provenance/upstream.json` pins the upstream repositories and revisions. The source
manifest records original Git blob IDs and SHA256 for every included file plus
explicit omitted paths. Source bytes are never edited during bundling.

The simulator's meshes, textures and state files are shipped in the upstream
LIBERO hierarchy because they are required runtime inputs. Third-party asset
notices embedded in that hierarchy are retained.

The 240 reference RGB frames originate from LIBERO-10 training demonstrations.
They are not extracted from the selected evaluation runs. Their selection and
record/frame provenance is retained in the reference map. Model checkpoints and
the full training dataset are not distributed here.
