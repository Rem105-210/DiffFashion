# DiffFashion

DiffFashion is a method for Fashion Design with a referenced natural image.

![images](IMG/framework.png)

## Getting Started

**Environment**

```bash
$ conda create --name DiffFashion python=3.9
$ conda activate DiffFashion
$ pip install ftfy regex matplotlib lpips kornia opencv-python torch==1.13.1+cu111 torchvision==0.14.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install color-matcher
$ pip install git+https://github.com/openai/CLIP.git
```

**Model download**

Download DDPM model pretrained on ImageNet from [model](https://github.com/openai/guided-diffusion)
Put the pt file under the project root folder as "./256x256_diffusion.pt"

**Model Run**

```
python main.py -i "input_example/bag1.jpg" --output_path "outputs/outputfile" -tg "input_example/fish1.jpg" --use_range_restart --diff_iter 100 --timestep_respacing 200 --skip_timesteps 80 --use_colormatch --use_noise_aug_all
```

## Example

![images](IMG/title.png)

![images](IMG/res.png)



