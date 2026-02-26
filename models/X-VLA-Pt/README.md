---
license: apache-2.0
base_model:
- microsoft/Florence-2-large
tags:
- robotics
- vla
pipeline_tag: robotics
---

#  X-VLA 0.9B (Foundation Edition)


**Repository:** [2toINF/X-VLA](https://github.com/2toinf/X-VLA)

**Authors:** [2toINF](https://github.com/2toINF) | **License:** Apache 2.0

**Paper:** *Zheng et al., 2025, “X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model”* ([arXiv:2510.10274](https://arxiv.org/pdf/2510.10274))


## 🚀 Overview

Successful generalist **Vision-Language-Action (VLA)** models rely on effective training across diverse robotic platforms with large-scale, cross-embodiment, heterogeneous datasets.
To facilitate and leverage the heterogeneity in rich robotic data sources, **X-VLA** introduces a **Soft Prompt approach** with minimally added parameters: we infuse prompt-learning concepts into cross-embodiment robot learning, introducing **separate sets of learnable embeddings** for each distinct embodiment.

These embodiment-specific prompts empower VLA models to exploit cross-embodiment features effectively.
Our architecture—**a clean, flow-matching-based VLA design relying exclusively on soft-prompted standard Transformers**—achieves superior scalability and simplicity.

Trained on **Bridge Data** and evaluated across **six simulations** and **three real-world robots**, the 0.9B-parameter X-VLA simultaneously achieves **state-of-the-art performance** across diverse benchmarks, demonstrating flexible dexterity and fast adaptation across embodiments, environments, and tasks.

🌐 **Project Website:** [https://thu-air-dream.github.io/X-VLA/](https://thu-air-dream.github.io/X-VLA/)


<video controls autoplay loop muted playsinline width="720">
  <source src="https://huggingface.co/2toINF/X-VLA-0.9B-WidowX/resolve/main/demo.mp4" type="video/mp4">
</video>

## ⚙️ Usage
### 🔹 Load the model

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "2toINF/X-VLA-WidowX",
    trust_remote_code=True
)
```
### 🔹 Start FastAPI server

```python
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("2toINF/X-VLA-WidowX", trust_remote_code=True)
model.run(processor, host="0.0.0.0", port=8000)
```
### 🔹 Client-server evaluation

You can run the provided evaluation client from our GitHub:
👉 [2toINF/X-VLA – Client &amp; Server Code](https://github.com/2toINF/X-VLA)


## 🧩 Architecture

| Component                         | Role                                                                       |
| :-------------------------------- | :------------------------------------------------------------------------- |
| **Florence 2 Encoder**      | Vision-Language representation backbone (encoder-only).                    |
| **SoftPromptedTransformer** | Flow-matching action denoiser using learnable soft prompts per embodiment. |
| **Action Hub**              | Defines action spaces, masking rules, pre/post-processing, and losses.     |

## 🧠 Training Summary

| Setting           | Value                                           |
| :---------------- | :---------------------------------------------- |
| Training Data     | Heterogeneous Datasets    |
| Parameters        | ≈ 0.9 B                                        |
| Action Mode       | `ee6d`                                         |
| Precision         | BP16                                            |
| Framework         | PyTorch + Transformers                          |

---
## 🪪 License
```
Copyright 2025 2toINF (https://github.com/2toINF)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
http://www.apache.org/licenses/LICENSE-2.0
```
---
## 📚 Citation
```bibtex
@article{zheng2025x,
  title   = {X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model},
  author  = {Zheng, Jinliang and Li, Jianxiong and Wang, Zhihao and Liu, Dongxiu and Kang, Xirui
             and Feng, Yuchun and Zheng, Yinan and Zou, Jiayin and Chen, Yilun and Zeng, Jia and others},
  journal = {arXiv preprint arXiv:2510.10274},
  year    = {2025}
}
```
---
## 🌐 Links

- 📄 **Paper:** [arXiv 2510.10274](https://arxiv.org/abs/2510.10274)
- 💻 **Code & Client/Server:** [GitHub – 2toINF/X-VLA](https://github.com/2toINF/X-VLA)
- 🤖 **Model Hub:** [Hugging Face – 2toINF/X-VLA-0.9B-WidowX](https://huggingface.co/2toINF/X-VLA-0.9B-WidowX)