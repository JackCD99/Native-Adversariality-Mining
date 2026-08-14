# EndoScene (evaluation only)

EndoScene is an unseen acquisition-center test set in the paper. Obtain it from the [EndoScene/CVC project](https://www.cvc.uab.es/CVC-Colon/index.php/cvc-endoscenestill/), retain its 60 image-mask pairs, convert masks to binary, and add relative paths to `test.list`. It must never appear in generator, miner, or downstream training.

Factory: `nam.data.endoscene.dataset:build_dataset`; only `split=test` is accepted.
