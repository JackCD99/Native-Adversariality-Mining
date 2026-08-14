# NAM exposure-mitigation strategies

The package provides four sampling-time strategies. The generator, downstream anchor, and trained
NAM miner remain frozen.

- `hat.py` replaces samples above a calibrated adversariality percentile and retains the lowest-loss
  retry when the trial budget is exhausted.
- `qsf.py` evaluates a condition-contour overlay with a configurable VQA scorer and acceptance
  threshold.
- `lsrs.py` compares full, unconditional, and component-conditioned predictions over cached DDIM
  states.
- `asg.py` aligns target-token cross-attention to the condition mask and penalizes self-attention
  conflicts at configured DDIM steps.

Each generator supplies a `NAMMitigationBackend`. HAT and QSF use the shared adapter bridge. LSRS and
ASG add architecture-specific state and attention access within the method package. Unsupported
capabilities fail explicitly.

References:

- [VQAScore](https://github.com/linzhiqiu/t2v_metrics), commit `6ecb74f92028f42c7e64546d3a71e98c8c73068f`.
- [CompLift](https://github.com/rainorangelemon/complift), commit `5b2158b6d91b20333af61ab56379bc3d551e3d30`.
- [InitNO](https://github.com/xiefan-guo/initno), commit `fa021c4881f6ac6869e399c032b13842fe3f9394`.
- [MedGemma](https://huggingface.co/google/medgemma-4b-it).
