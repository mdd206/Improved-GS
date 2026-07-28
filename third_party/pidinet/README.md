# PiDiNet inference dependency

This directory contains the minimum PiDiNet inference code and official
`table5_pidinet.pth` checkpoint needed by the HF-GS edge prior.

- Upstream: https://github.com/hellozhuo/pidinet
- Upstream revision inspected: `d21aa881ed9c628571636fad39acfe1fad517ebd`
- Architecture: full PiDiNet, CARv4, spatial attention, dilation 24
- Checkpoint SHA-256:
  `80860ac267258b5f27486e0ef152a211d0b08120f62aeb185a050acc30da486c`
- Input preprocessing: RGB in `[0,1]`, ImageNet mean/std normalization
- Output: the final fused sigmoid edge map

The upstream license includes an additional research-purpose condition. Read
[LICENSE](LICENSE) before using this dependency outside research.
