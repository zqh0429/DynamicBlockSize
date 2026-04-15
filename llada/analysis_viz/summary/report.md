# Score Analysis Report

- Number of samples: 8
- Global combined-score mean: 0.001363
- Global cut local-rank mean: 0.9180
- Block size histogram: {5: 1, 7: 1, 25: 1, 27: 1, 32: 126}

## Sample Summary

- sample_0001: blocks=[32, 32, 32, 32, 32, 32, 32, 25, 32, 32, 32, 32, 32, 32, 32, 32, 7] | avg_block=30.118 | combined_mean=0.001497 | cut_mean=0.003756
- sample_0002: blocks=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32] | avg_block=32.000 | combined_mean=0.001277 | cut_mean=0.003138
- sample_0003: blocks=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32] | avg_block=32.000 | combined_mean=0.001522 | cut_mean=0.003373
- sample_0004: blocks=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32] | avg_block=32.000 | combined_mean=0.001434 | cut_mean=0.003104
- sample_0005: blocks=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32] | avg_block=32.000 | combined_mean=0.001526 | cut_mean=0.003492
- sample_0006: blocks=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32] | avg_block=32.000 | combined_mean=0.001387 | cut_mean=0.003881
- sample_0007: blocks=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32] | avg_block=32.000 | combined_mean=0.001368 | cut_mean=0.003229
- sample_0008: blocks=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 27, 32, 32, 5] | avg_block=30.118 | combined_mean=0.000896 | cut_mean=0.002540

## Structural Token Lift

- newline: mean=0.001357 | lift=0.995 | count=327
- punct: mean=0.001398 | lift=1.025 | count=342
- keyword: mean=0.001452 | lift=1.065 | count=114
- fence: mean=0.001271 | lift=0.932 | count=18
- indent: mean=0.001204 | lift=0.883 | count=36
- other: mean=0.001360 | lift=0.997 | count=3259

## Best Layers For Boundary Detection

- layer 1: peak@cut=1.000 | peak@cut±1=1.000 | cut/neighbor=4.983 | local_rank=1.000
- layer 4: peak@cut=0.621 | peak@cut±1=1.000 | cut/neighbor=3.223 | local_rank=0.923
- layer 5: peak@cut=0.902 | peak@cut±1=0.992 | cut/neighbor=3.409 | local_rank=0.977
- layer 6: peak@cut=0.960 | peak@cut±1=0.984 | cut/neighbor=4.305 | local_rank=0.990
- layer 11: peak@cut=0.878 | peak@cut±1=0.983 | cut/neighbor=4.759 | local_rank=0.972
- layer 9: peak@cut=0.688 | peak@cut±1=0.983 | cut/neighbor=3.570 | local_rank=0.933
- layer 28: peak@cut=0.870 | peak@cut±1=0.975 | cut/neighbor=5.814 | local_rank=0.971
- layer 29: peak@cut=0.747 | peak@cut±1=0.975 | cut/neighbor=4.755 | local_rank=0.943
- layer 8: peak@cut=0.951 | peak@cut±1=0.975 | cut/neighbor=4.991 | local_rank=0.988
- layer 0: peak@cut=0.933 | peak@cut±1=0.958 | cut/neighbor=1.345 | local_rank=0.978
- layer 15: peak@cut=0.722 | peak@cut±1=0.928 | cut/neighbor=3.686 | local_rank=0.930
- layer 14: peak@cut=0.781 | peak@cut±1=0.928 | cut/neighbor=3.461 | local_rank=0.943
