import os
from pathlib import Path
from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

from utils_visualize.metrics_accumulator import MetricsAccumulator
from utils_visualize.video import save_video

from numpy import random
from optimization.augmentations import ImageAugmentations

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from optimization.losses import range_loss, d_clip_loss
import lpips
import numpy as np
from src.vqc_core import *
from model_vit.loss_vit import Loss_vit
from CLIP import clip
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    create_classifier,
    classifier_defaults,
)
import collections
from utils_visualize.visualization import show_tensor_image, show_editied_masked_image
from pathlib import Path
from id_loss import IDLoss

from color_matcher import ColorMatcher
from color_matcher.io_handler import load_img_file, save_img_file, FILE_EXTS
from color_matcher.normalizer import Normalizer

mean_sig = lambda x:sum(x)/len(x)


class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps


def l2_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def cos_layers(xs_conv_features, ys_conv_features, clip_model_name):
    if "RN" in clip_model_name:
        return [torch.square(x_conv, y_conv, dim=1).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]
    return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


class CLIPConvLoss(torch.nn.Module):
    def __init__(self, device):
        super(CLIPConvLoss, self).__init__()
        self.clip_model_name = "ViT-B/32"
        self.clip_conv_loss_type = "L2"
        self.clip_fc_loss_type = "Cos"  # args.clip_fc_loss_type

        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        self.model, clip_preprocess = clip.load(
            self.clip_model_name, device, jit=False)
        # 声明CLIP model
        self.visual_encoder = CLIPVisualEncoder(self.model)
        # 在这里 应该是CLIPVisualEncoder

        self.img_size = 224
        self.model.eval()
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()
        self.device = device


        self.clip_fc_layer_dims = None  # self.args.clip_fc_layer_dims
        self.clip_conv_layer_dims = None  # self.args.clip_conv_layer_dims
        self.clip_fc_loss_weight = 0.5
        self.counter = 0

    def forward(self, sketch, target, style,list1,list2,can1,can2,mode="train"):
        # skecth是当前image
        # target是init_image
        # style是style_image
        # list1用来控制和init_image的相似度
        # list2用来控制和style的相似度
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        #         y = self.target_transform(target).to(self.args.device)
        conv_loss_dict = {}
        style = style.to(self.device)
        x = sketch.to(self.device)
        y = target.to(self.device)

        sketch_augs, img_augs,i_style = [self.normalize_transform(x)], [
            self.normalize_transform(y)],[self.normalize_transform(style)]
        # 对输入图进行变换 使得能够输入
        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)
        ss = torch.cat(i_style,dim=0).to(self.device)


        xs_fc_features, xs_conv_features = self.visual_encoder(xs)
        with torch.no_grad():
            ys_fc_features, ys_conv_features = self.visual_encoder(ys)
            style_fc, ss_conv_features = self.visual_encoder(ss)

        loss1 = 0.0
        conv_loss_init = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ys_conv_features, self.clip_model_name)
        conv_loss_style = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ss_conv_features, self.clip_model_name)

        for i in range(12):
            loss1 += conv_loss_init[i] * list1[i]
        for i in range(12):
            loss1 += conv_loss_style[i] * list2[i]
        if self.clip_fc_loss_weight:
            # fc distance is always cos
            fc_loss_1 = (1 - torch.cosine_similarity(xs_fc_features,
                                                   ys_fc_features, dim=1)).mean()
            fc_loss_2 = (1 - torch.cosine_similarity(xs_fc_features,
                                                      style_fc,dim=1)).mean()

            conv_loss_dict["fc"] = fc_loss_1 * can1 * self.clip_fc_loss_weight + fc_loss_2 * can2 * self.clip_fc_loss_weight

        loss1 += conv_loss_dict["fc"]
        return loss1

class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        os.makedirs(self.args.output_path, exist_ok=True)

        self.ranked_results_path = Path(self.args.output_path)
        os.makedirs(self.ranked_results_path, exist_ok=True)

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults()
        use_class = True
        if self.args.use_ffhq:
            self.model_config.update(
            {
                "attention_resolutions": "16",
                "class_cond": use_class,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 128,
                "num_head_channels": 64,
                "num_res_blocks": 1,
                "resblock_updown": True,
                "use_fp16": False,
                "use_scale_shift_norm": True,
            }
        )
        else:
            self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": use_class,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )
        cls_config = dict(
            image_size=256,
            classifier_use_fp16=False,
            classifier_width=128,
            classifier_depth=2,
            classifier_attention_resolutions="32,16,8",  # 16
            classifier_use_scale_shift_norm=True,  # False
            classifier_resblock_updown=True,  # False
            classifier_pool="attention",
        )
        self.classigo = create_classifier(**cls_config)
        self.classigo.load_state_dict(torch.load("256x256_classifier.pt", map_location="cpu"))


        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)
        self.classigo.requires_grad_(False).eval().to(self.device)
        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)

        
        if self.args.use_ffhq:
            # false
            self.model.load_state_dict(
                torch.load(
                    "./checkpoints/ffhq_10m.pt",
                    map_location="cpu",
                )
            )
            self.idloss = IDLoss().to(self.device)
        else:
            self.model.load_state_dict(
            torch.load(
                "256x256_diffusion.pt"
                if self.args.model_output_size == 256
                else "checkpoints/512x512_diffusion.pt",
                map_location="cpu",
            )
        )
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()
        with open("model_vit/config.yaml", "r") as ff:
            config = yaml.safe_load(ff)

        cfg = config
        
        self.VIT_LOSS = Loss_vit(cfg, lambda_ssim=self.args.lambda_ssim,lambda_dir_cls=self.args.lambda_dir_cls,lambda_contra_ssim=self.args.lambda_contra_ssim,lambda_trg=args.lambda_trg).eval()#.requires_grad_(False)
        names = ['ViT-B/32']
        # init networks
        self.clip_model = CLIPConvLoss(self.device)
        # init networks
        if self.args.target_image is None:
            self.clip_net = CLIPS(names=names, device=self.device, erasing=False)#.requires_grad_(False)
        self.cm = ColorMatcher()
        self.clip_size = 224
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        #   self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

        self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()
    def noisy_aug(self,t,x,x_hat):
        fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
        x_mix = x_hat * fac + x * (1 - fac)
        return x_mix
    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()
        return unscaled_timestep

    def edit_image_by_prompt(self):
        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.our_parameter = {
            'pa1': 0,
            # CLIP最后的vector 对齐 目前图片 和
            'pa2': 0,
            'palist1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'palist2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'para_vit': 1,
            'p_vit_vector': 0.5,
            'use_mask': True,
            'iterratio': 0,
            'iterlong': 36,
            'dis_type': 'l2',
            'the': -0.2,
            'mix': 1,
            'use_part': True,
        }

        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
        self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_image = (
            TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )

        self.target_image = None
        if self.args.target_image is not None:
            self.target_image_pil = Image.open(self.args.target_image).convert("RGB")
            self.target_image_pil = self.target_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
            self.target_image = (
                TF.to_tensor(self.target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
            )
        self.prev = self.init_image.detach()
        # text guide
        self.flag_resample=False
        total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1

        # 很垃圾的对海洋风格分类

        def cond_fn(x, t, y=None):
            # 更新梯度
            if self.args.prompt == "":
                return torch.zeros_like(x)
            self.flag_resample = False
            with torch.enable_grad():
                frac_cont = 1.0
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )
                
                loss = torch.tensor(0)


                if self.args.use_noise_aug_all:
                    x_in = self.noisy_aug(t[0].item(),x,out["pred_xstart"])
                else:
                    x_in = out["pred_xstart"]

                
                if self.args.vit_lambda != 0:

                    if t[0] > self.args.diff_iter:
                        vit_loss, vit_loss_val = self.VIT_LOSS(x_in,self.init_image,self.prev,use_dir=True,frac_cont=frac_cont,target = self.target_image,target_ca=self.our_parameter['p_vit_vector'],mask=self.mask)
                    else:
                        vit_loss, vit_loss_val = self.VIT_LOSS(x_in,self.init_image,self.prev,use_dir=False,frac_cont=frac_cont,target = self.target_image,target_ca=self.our_parameter['p_vit_vector'],mask=self.mask)
                    loss = loss + self.our_parameter['para_vit'] * vit_loss

                    
                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                if self.target_image is not None:
                    loss = loss + mse_loss(x_in, self.target_image) * self.args.l2_trg_lambda


                Closs = self.clip_model(0.5 * x_in + 0.5, 0.5 * self.init_image + 0.5,
                                        0.5 * self.target_image + 0.5,self.our_parameter['palist1'],
                                        self.our_parameter['palist2'],self.our_parameter['pa1'],self.our_parameter['pa2'])
                loss = loss + Closs

                
                if self.args.use_ffhq:
                    loss = loss + self.idloss(x_in,self.init_image) * self.args.id_lambda
                self.prev = x_in.detach().clone()
                
            return -torch.autograd.grad(loss, x)[0], self.flag_resample

        # classifier的推理部分
        save_image_interval = self.diffusion.num_timesteps // 5

        # section 1 得到包的mask
        # 要想加classcondition
        # 需要调用p_sample的时候再调用p_mean_variance传入参数给Unet的时候写上第三个参数即可

        # y: an [N] Tensor of labels, if class-conditional 传入模型的时候
        # 需要传入到model_kwargs
        # 用key值y即可
        # 636 n03709823 包, mailbag, postbag
        mask_func = (
                self.diffusion.ddim_sample_loop_progressive
                if self.args.ddim
                else self.diffusion.p_sample_loop_progressive
        )
        sample_masks = mask_func(
            self.model,
            (
                self.args.batch_size,
                3,
                self.model_config["image_size"],
                self.model_config["image_size"],
            ),
            clip_denoised=False,
            model_kwargs={"y": torch.Tensor([636]).long().to(self.device)},
            # 假设了batch_size == 1
            cond_fn=None,
            # 无须classifier去修正梯度
            progress=True,
            skip_timesteps=self.args.skip_timesteps,
            init_image=self.init_image,
            # 这里规定了初始的init_image
            # 在这里传入
            postprocess_fn=None,
            randomize_class=False,
        )
        total_steps_with_resample = self.diffusion.num_timesteps - self.args.skip_timesteps - 1 + (
                    self.args.resample_num - 1)
        print(total_steps_with_resample)
        mask = None
        for step, mask_sample in enumerate(sample_masks):

            if step == total_steps_with_resample/2:
                # in 50% steps
                for b in range(self.args.batch_size):
                    yuce = mask_sample['sample'][0].add(1).div(2).clamp(0, 1)
                    pil_y = TF.to_pil_image(yuce)
                    pil_y.save('outputs/onlyclip/masklll.png')
                    
                    visualization_path = Path(
                        os.path.join(self.args.output_path, self.args.output_file)
                    )
                    visualization_path = visualization_path.with_name(
                        f"{visualization_path.stem}_b_masksksksksk_{b}{visualization_path.suffix}"
                    )
                    ranked_pred_path = self.ranked_results_path / (visualization_path.name)
                    
                    mask = self.diffusion.get_mask(self.model, mask_sample["pred_xstart"], t=step-1, device=self.device,losstype=self.our_parameter['dis_type'],sigma=self.our_parameter['the'])
                    mask_image = mask.add(1).div(2).clamp(0, 1)
                    #  mask_sample["pred_xstart"]
                    pred_image_pil = TF.to_pil_image(mask_image)
                    pred_image_pil.save(ranked_pred_path)

            # ranked_pred_path = self.ranked_results_path / (visualization_path.name)
        # end section 1
        if self.our_parameter['use_part']:
            self.mask = mask
        else:
            self.mask = None
        print("end section 1")

        # 200 - 40 origin=1000
        total_steps_with_resample = self.diffusion.num_timesteps - self.args.skip_timesteps - 1 + (
                self.args.resample_num - 1)
        for iteration_number in range(self.args.iterations_num):
            # num = 10 生成10次图片
            # 也就是重复推理10次
            sample_func = (
                self.diffusion.ddim_sample_loop_progressive
                if self.args.ddim
                else self.diffusion.h_sample_loop_progressive
            )
            # 获得diffusion的采样函数，来获得sample，也就是一次运行,
            # 其中获取下面的参数之后，会序列化yield模型，来源源不断生成样本
            # 这些所有的方法都bind在GaussianDiffusion这个大类框架下
            # 这时把另一个东西model传进去调用方法
            samples = sample_func(
                self.model,
                (
                    self.args.batch_size,
                    3,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                mask=mask,
                clip_denoised=False,
                model_kwargs={"y": torch.Tensor([636]).long().to(self.device)},
                cond_fn=cond_fn,
                # 修正梯度
                progress=True,
                skip_timesteps=self.args.skip_timesteps,
                init_image=self.target_image,
                # 这里规定了初始的init_image
                # 在这里传入
                bag_image=self.init_image,
                use_mask=self.our_parameter['use_mask'],
                iterratio=self.our_parameter['iterratio'],
                dual_iter=self.our_parameter['iterlong'],
                mixup= self.our_parameter['mix'],
                # 原始的bag图的混合占比
                postprocess_fn=None,
                randomize_class=True,
                classi=self.classigo,
                vit_func=self.VIT_LOSS
            )
            if self.flag_resample:
                continue
            
            intermediate_samples = [[] for i in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1 
            total_steps_with_resample = self.diffusion.num_timesteps - self.args.skip_timesteps - 1 + (self.args.resample_num-1)
            for j, sample in enumerate(samples):
                should_save_image = j % save_image_interval == 0 or j == total_steps_with_resample
                # 最后的条件是结束

                # self.metrics_accumulator.print_average_metric()
                if j == 60:
                    s_image = sample["pred_xstart"][0]
                    sl_image = s_image.add(1).div(2).clamp(0, 1)
                    sl_image_pil = TF.to_pil_image(sl_image)
                    sl_image_pil.save('outputs/onlyclip/xstart.png')
                    now_image = sample['sample'][0]
                    now_image = now_image.add(1).div(2).clamp(0, 1)
                    now_pil = TF.to_pil_image(now_image)
                    now_pil.save('outputs/onlyclip/now20.png')
                    j_image = sample["pred_xstart"][0] * mask
                    sj_image = j_image.add(1).div(2).clamp(0, 1)
                    sj_image_pil = TF.to_pil_image(sj_image)
                    sj_image_pil.save('outputs/onlyclip/xjjjjjjjj.png')


                for b in range(self.args.batch_size):
                    pred_image = sample["pred_xstart"][b]
                    visualization_path = Path(
                        os.path.join(self.args.output_path, self.args.output_file)
                    )
                    visualization_path = visualization_path.with_name(
                        f"{visualization_path.stem}_i_{iteration_number}_b_{b}{visualization_path.suffix}"
                    )

                    pred_image = pred_image.add(1).div(2).clamp(0, 1)
                    pred_image_pil = TF.to_pil_image(pred_image)
            ranked_pred_path = self.ranked_results_path / (visualization_path.name)
            
            if self.args.target_image is not None:
                if self.args.use_colormatch:
                    src_image = Normalizer(np.asarray(pred_image_pil)).type_norm()
                    trg_image = Normalizer(np.asarray(self.target_image_pil)).type_norm()
                    img_res = self.cm.transfer(src=src_image, ref=trg_image, method='mkl')
                    img_res = Normalizer(img_res).uint8_norm()
                    save_img_file(img_res, str(ranked_pred_path))
            else:
                pred_image_pil.save(ranked_pred_path)


