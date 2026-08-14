# Synapse Multi-Organ

Download the 30 contrast-enhanced abdominal CT scans from [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) after accepting its access terms. NAM uses background plus spleen, right kidney, left kidney, gallbladder, esophagus, liver, stomach, aorta, and pancreas.

## NAM preparation

Create a patient-level 7:1:2 split with seed 42. Convert volumes to axial slices, remove slices with no foreground organ, retain the ten-class label convention (background plus nine organs), and resize to 256 x 256. The paper reports 3,779 foreground slices: 2,645/378/756 for train/validation/test. Put processed arrays below `data/` and record portable relative paths in each list.

```text
case0001_slice042 data/images/case0001_slice042.npy data/labels/case0001_slice042.npy
```

Factory: `nam.data.synapse.dataset:build_dataset`.
