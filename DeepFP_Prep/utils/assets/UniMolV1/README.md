---
language: en
license: mit
tags:
- Uni-Mol
- Pretrained Models
---

## Checkpoints for DeepFP_Prep

Place **Uni-Mol V1** pretraining weights in this directory (see `env_utils.py`):

| File | Download |
|------|----------|
| `mol_pre_no_h_220816.pt` (typical default) | [mol_pre_no_h_220816.pt](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol_pre_no_h_220816.pt) |
| `mol_pre_all_h_220816.pt` (optional) | [mol_pre_all_h_220816.pt](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol_pre_all_h_220816.pt) |
| `mol.dict.txt` | Often already in repo; else [mol.dict.txt](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol.dict.txt) |

Index: `../WEIGHTS_DOWNLOAD_LIST.txt`.

# Huggingface Model Repository

Welcome to our Huggingface model repository, where we specialize in models pretrained with Uni-Mol for various applications. Uni-Mol, A Universal 3D Molecular Representation Learning Framework.

## Models Overview

Our models, pretrained with Uni-Mol, are designed to excel in diverse environments. Here's a brief overview of the scenarios where they can be applied:

- **mol_pre_all_h_220816.pt**: This model is a pre-trained version that includes hydrogen atoms in its training data. It's designed for applications requiring detailed molecular structures, including hydrogen atoms, to predict molecular properties accurately.

- **mol_pre_no_h_220816.pt**: A counterpart to the above, this model is pre-trained without hydrogen atoms in its training data. It's optimized for scenarios where the presence of hydrogen atoms is assumed or irrelevant, focusing on the core molecular structure for predictions.

- **mp_all_h_230313.pt**: This model is trained using data from the Materials Project, and is specialized for predicting properties of crystalline materials. It's ideal for applications in materials science, where understanding the properties of materials at the atomic level is crucial.

- **oled_pre_no_h_230101.pt**: Specifically trained on data related to optoelectronic molecules without including hydrogen atoms in the training data. This model is tailored for predicting properties of molecules used in OLED (Organic Light Emitting Diode) technologies, focusing on core structures that influence optoelectronic behaviors.

- **poc_pre_220816.pt**: This model is trained on 'protein pocket' related data, focusing on the interactions within molecular pockets. It's particularly useful for applications in drug discovery and enzyme activity prediction, where understanding the molecular interactions within pockets is essential.

Additionally, for each `.pt` model file, there is a corresponding `*_dict.txt` file. These dictionary files are crucial as they map the model's features, ensuring accurate data interpretation and prediction.

## GitHub Repository

For more detailed information about our models, including how to download, install, and use them in your projects, please visit our GitHub repository:

[Uni-Mol Repository](https://github.com/deepmodeling/Uni-Mol)

For comprehensive documentation, including tutorials and API references, please visit our documentation site:

[Uni-Mol tools documentation](https://unimol.readthedocs.io/en/latest/)

## Getting Started

To get started with our Uni-Mol pretrained models, you can follow the instructions in our GitHub repository. We provide comprehensive guides and examples to help you integrate these models into your applications seamlessly.

## Support

If you encounter any issues or have questions, feel free to open an issue on our GitHub repository, and we'll be happy to assist you.

Thank you for choosing our Huggingface models pretrained with Uni-Mol. We're excited to see the incredible applications you'll build with them!

## License

This project is licensed under the MIT License

