# CVC-ColonDB (evaluation only)

CVC-ColonDB supplies 380 unseen colonoscopy image-mask pairs for acquisition-shift evaluation. Obtain it from the CVC polyp dataset distribution, retain all 380 samples, convert masks to binary, and populate `test.list` with relative paths. Do not merge these cases into the Polyps training benchmark.

Factory: `nam.data.colondb.dataset:build_dataset`; only `split=test` is accepted.
