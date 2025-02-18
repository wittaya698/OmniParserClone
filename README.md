# OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent

<p align="center">
  <img src="imgs/logo.png" alt="Logo">
</p>

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OmniParser** is a comprehensive method for parsing user interface screenshots into structured and easy-to-understand elements, which significantly enhances the ability of GPT-4V to generate actions that can be accurately grounded in the corresponding regions of the interface.

## Install

Install environment:

```python
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

Then download the model ckpts files in: https://huggingface.co/microsoft/OmniParser, and put them under weights/, default folder structure is: weights/icon_detect, weights/icon_caption_florence, weights/icon_caption_blip2.

For v1:
convert the safetensor to .pt file.

```python
python weights/convert_safetensor_to_pt.py

For v1.5:
download 'model_v1_5.pt' from https://huggingface.co/microsoft/OmniParser/tree/main/icon_detect_v1_5, make a new dir: weights/icon_detect_v1_5, and put it inside the folder. No weight conversion is needed.
```

## Examples:

We put together a few simple examples in the demo.ipynb.

## Gradio Demo

To run gradio demo, simply run:

```python
# For v1
python gradio_demo.py --icon_detect_model weights/icon_detect/best.pt --icon_caption_model florence2
# For v1.5
python gradio_demo.py --icon_detect_model weights/icon_detect_v1_5/model_v1_5.pt --icon_caption_model florence2
```

## Model Weights License

For the model checkpoints on huggingface model hub, please note that icon_detect model is under AGPL license since it is a license inherited from the original yolo model. And icon_caption_blip2 & icon_caption_florence is under MIT license. Please refer to the LICENSE file in the folder of each model: https://huggingface.co/microsoft/OmniParser.

## Important Notice

⚠️ **This project is intended for research and educational purposes only.**
