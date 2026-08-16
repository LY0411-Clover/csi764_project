| Dataset         | Domain                           |
| --------------- | -------------------------------- |
| seq-cifar10     | natural images (small)           |
| seq-cub200      | fine-grained birds               |
| seq-eurosat-rgb | satellite imagery                |
| seq-imagenet-r  | natural image distribution shift |
| seq-tinyimg     | large-scale object images        |

| Dataset         | 类型                      | 难度              | 对 LEAR 意义                 |
| --------------- | ------------------------- | ----------------- | ---------------------------- |
| seq-cifar10     | natural image             | 简单              | sanity check                 |
| seq-eurosat-rgb | remote sensing            | domain shift 明显 | 测 domain continual learning |
| seq-imagenet-r  | image style shift         | 较难              | 测 expert reuse              |
| seq-tinyimg     | large-scale natural image | 较难              | 测扩展能力                   |
