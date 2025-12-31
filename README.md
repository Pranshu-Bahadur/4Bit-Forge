![4Bit-Forge logo](assets/4Bit-Forge.png)

This repo builds upon the solid foundation laid in MoE-Quant. Thank you for your amazing work!

Todo(s):
- [x] io.py + handle quantized layers
- [x] scaffold initial quant.py
- [ ] engine.py
- [ ] preprocess.py (Math500)
    - [ ] use chat template provided by model files
- [ ] pack uint8 <- int4?
- [ ] sparsegptq 2:4 vllm inference compatible format
    - [ ] build interblock Mask update sparsegptq 2:4 kernel
    - [ ] Test sparsegptq on gpt 120B oss ("mvp")

- [ ] metrics.py / stat view during compression run (?)
- [ ] multigpu support
- [ ] R&D
    - [ ] spargptq 1:8
    - [ ] Custom Sparse 1:8 Kernels for inference
    - [ ] Monkey patch with transformers/torch nn.Linear
    - [ ] Explore monkey patching with vllm