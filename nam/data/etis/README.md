# ETIS-LaribPolypDB (evaluation only)

ETIS contains 196 unseen colonoscopy image-mask pairs used only for acquisition-shift testing. Download it through the ETIS-Larib polyp benchmark distribution, convert masks to binary, and populate `test.list`. These samples must remain isolated from training and validation.

Factory: `nam.data.etis.dataset:build_dataset`; only `split=test` is accepted.
