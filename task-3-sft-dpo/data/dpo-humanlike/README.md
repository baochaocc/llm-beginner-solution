---
language:
- en
license: llama3
configs:
- config_name: default
  data_files:
  - split: train
    path: data.json
---

# 本数据集论文解读，请看公众号文章 👇🏻

### <img src="https://www.modelscope.cn/datasets/okwinds/Human-Like-DPO-Dataset/resolve/master/wechat.png" width="30" height="30" align="absmiddle"> 觉察流 - [AI的“人味儿”从何而来？DPO和LoRA打造更拟人化的AI](https://mp.weixin.qq.com/s/59WEBKi0uGYCwOXsd5FgCw)

<br/>

# 下载方法 
:modelscope-code[]{type="sdk"}
:modelscope-code[]{type="git"}


> <span style="color:red;font-size:16px"> 声明：本数据集完全转载自 Huggingface 上的 [HumanLLMs/Human-Like-DPO-Dataset](https://huggingface.co/HumanLLMs/Human-Like-DPO-Dataset) <br/>更多模型信息，请关注下文👇🏻， 为原数据集仓库的中文版说明。</span>

<br/>

#### _仓库作者在此 👇🏻 扫一扫_

<img src="https://www.modelscope.cn/models/okwinds/GPT-2/resolve/master/qrcode_for_jcl_258.jpg" />

_______________________________

<br/>
<br/>

<div align="center">
  <img src="https://www.modelscope.cn/datasets/okwinds/Human-Like-DPO-Dataset/resolve/master/avatar.jpeg" width="320" height="320" />
  <h1>提升大型语言模型中的拟人化响应</h1>
</div>

<p align="center">
  &nbsp&nbsp | 🤖 <a href="https://www.modelscope.cn/collections/Human-Like-nirenyingda-38b077cf6d0a44">模型集合</a>&nbsp&nbsp | 
  &nbsp&nbsp 📊 <a href="https://www.modelscope.cn/datasets/okwinds/Human-Like-DPO-Dataset">数据集</a>&nbsp&nbsp | 
  &nbsp&nbsp <img src="https://www.modelscope.cn/datasets/okwinds/Human-Like-DPO-Dataset/resolve/master/wechat.png" width="22" height="22" align="absmiddle"> <a href="https://mp.weixin.qq.com/s/59WEBKi0uGYCwOXsd5FgCw">论文解读</a>&nbsp&nbsp | 
  &nbsp&nbsp 📄<a href="https://arxiv.org/abs/2501.05032">论文</a>&nbsp&nbsp |
</p>


# **Human-Like-DPO-Dataset**

此数据集是作为旨在提高大型语言模型对话流畅度和参与度研究的一部分创建的。它适用于直接偏好优化（DPO）等格式，以引导模型生成更符合人类语言风格的响应。

本数据集包含 **256 主题** 中的 **10,884 样本**，囊括了:

- 技术
- 日常生活
- 科学
- 历史
- 艺术

每个样本包含：
- **对话问题**: 自然、吸引人的问题，反映日常人类对话。
- **拟人回应**: 生成自然、对话式的答案，以模仿人类互动。
- **正式回复**: 反映传统人工智能响应的结构化、专业回答。

# **数据集用途**
此数据集可用于微调LLM：
- 提高对话连贯性。
- 减少机械或非个性化响应。
- 提升对话系统的情商。

更多关于数据集创建和使用的详细信息可以在随附的研究论文中找到。 <img src="https://www.modelscope.cn/datasets/okwinds/Human-Like-DPO-Dataset/resolve/master/wechat.png" width="22" height="22" align="absmiddle"> [论文解读](https://mp.weixin.qq.com/s/59WEBKi0uGYCwOXsd5FgCw) | 📄[研究论文](https://arxiv.org/abs/2501.05032).
