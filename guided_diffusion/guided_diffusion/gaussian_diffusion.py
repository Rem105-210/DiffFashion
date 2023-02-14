"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):

    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = enum.auto()  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self, *, betas, model_mean_type, model_var_type, loss_type, rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        # 用Unet去从xt和timestamp来得到xt-1的均值和方差
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        # var type是LEARN_RANGE
        # mean type是EPSILON
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # 如果定义的方差类型是可学习的
            # 两种，一个是直接去预测方差，一个是去预测方差的范围（iddpm）
            # 如果这样的话，通道数是翻倍的C（均值）--->2C（均值和方差）
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:

                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
                # 预测对数方差然后直接取指数
            else:
                # 预测方差的范围
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            #对于不可学习的方差来说
            # 需要我们从以往的b中来收集出来第t时刻的方差
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
            # 抽出来第t步的方差

        def process_xstart(x):
            # 对x进行一定的处理
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        # 来预测均值
        # 有三种 一个是预测噪声一个是预测xt-1一个是预测x0
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            # 直接去预测xt-1的u
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            # 还算了一个量，在训练时候不会用，后面的evaluate会用
            # 知道了xt和xt-1，来计算x0
            model_mean = model_output
            # 那么modelmean等于modeloutput
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                # 预测噪声，那么需要我们去预测一下x0，再由x0来计算噪声
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                    # 上面这个 函数是用来从xt和t来预测x0的
                )
            # 从x0,t,xt来得出xt-1的均值
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient,flag = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean,flag

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        # t = t.long()
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        # print(alpha_bar)
        # print(t)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        # print(self._scale_timesteps(t).dtype)
        gradient,flag = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        eps = eps - (1 - alpha_bar).sqrt() * gradient

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out,flag

    def get_mask(self, model, x, t, device, losstype, sigma,clip_denoised=False, denoised_fn=None, model_kwargs=None,):
        name_t = th.tensor([20] * 1, device=device)
        a = th.zeros_like(x)
        for i in range(5):
            out = self.p_mean_variance(
                model,
                x,
                name_t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs={"y": th.Tensor([636]).long().to(device)},
            )
            a = a + out['mean']/5

        other = th.zeros_like(x)
        candi = [10, 487, 561, 891, 721, 897, 102, 204, 338, 956]
        candi = [10, 561, 891, 721, 897, 102, 204, 338, 956]

        for i in range(9):
            op = {"y": th.Tensor([candi[i]]).long().to(device)}
            b = self.p_mean_variance(model, x, name_t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                     model_kwargs=op, )
            other = other + b['mean']/9
            temp1 = (b['mean'] - a).pow(2).sum(dim=1)
            import cv2
            reskk = np.array(temp1.cpu())
            reskk = cv2.GaussianBlur(reskk, (5, 5), 0)
            reskk = torch.from_numpy(reskk).to(device)
            temp = (reskk -reskk.mean())/reskk.std() > sigma
            from torchvision.transforms import functional as TF
            yuce =temp.add(1).div(2).clamp(0, 1)
            pil_y = TF.to_pil_image(yuce)
            pil_y.save(f'outputs/onlyclip2/masklll_{i}.png')
        if losstype == "l2":
            res = (a - other).pow(2).sum(dim=1)
        else:
            res = abs(a - other).sum(dim=1)

        mean = res.mean()
        std = res.std()
        return (res - mean) / std > sigma

    def p_dual_loop(
        self,
        model,
        shape,
        mask,
        bag_image,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
    ):
        final = None
        for sample in self.dual_sample_loop(
            model,
            shape,
            mask,
            bag_image,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
        ):
            final = sample
        return final["sample"]

    def dual_sample_loop(
            # 这里有bug
            # 呜呜呜
            self,
            model,
            shape,
            mask,
            bag_image,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            skip_timesteps=0,
            init_image=None,
            postprocess_fn=None,
            randomize_class=False,
            find_init=False
    ):
        # 去推理的一个过程
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        # 生成T时刻（最后一个时刻）一个标准的噪声

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)
        # 如果不输入init_image则代表让我们进行随机采样

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]
        # 得到一个time_step的时刻，然后将这个range进行逆序，从T..到0

        batch_size = shape[0]
        init_image_batch = th.tile(init_image, dims=(batch_size, 1, 1, 1))
        bag_image_batch_0 = th.tile(bag_image, dims=(batch_size, 1, 1, 1))

        img = self.q_sample(
            x_start=init_image_batch,
            t=th.tensor(indices[0], dtype=th.long, device=device),
            noise=img,
        )
        pre_image = img[0].add(1).div(2).clamp(0, 1)
        from torchvision.transforms import functional as TF
        pre_image_pil = TF.to_pil_image(pre_image)
        pre_image_pil.save('outputs/onlyclip/x0.png')
        n_add = th.randn(*shape, device=device)
        gai_img = self.q_sample(
            x_start=bag_image_batch_0,
            t=th.tensor(indices[0], dtype=th.long, device=device),
            noise=n_add,
        )
        gai = gai_img[0].add(1).div(2).clamp(0, 1)
        e_image_pil = TF.to_pil_image(gai)
        e_image_pil.save('outputs/onlyclip/yt.png')


        # 输入加的噪声XT，第一个时间戳，和原始图片
        # 得到的img是加噪后的图片
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        flag = False
        while True:
            if flag:
                indices = list(range(self.num_timesteps - skip_timesteps))[::-1]
                indices = tqdm(indices)
            image_after_step = img
            for i in indices:
                # 对这个时间进行遍历
                if flag:
                    # 除了步骤XT都进来
                    img = th.randn(*shape, device=device)
                    # 生成随机噪声
                    img = self.q_sample(
                        x_start=init_image_batch,
                        t=th.tensor([i] * shape[0], device=device),
                        noise=img,
                    )
                    image_after_step = img
                    # 生成这个timestep的前向加噪图（从后向前）
                if i == self.num_timesteps - skip_timesteps - 1:
                    # XT时候从这里
                    for r in range(10):
                        t = th.tensor([i] * shape[0], device=device)
                        # 循环10个timestep
                        if randomize_class and "y" in model_kwargs:
                            model_kwargs["y"] = th.randint(
                                low=0,
                                high=model.num_classes,
                                size=model_kwargs["y"].shape,
                                device=model_kwargs["y"].device,
                            )
                        # 如果是condition的，并且没有给condition的classlabel
                        # 那么我就自己随机生成一个
                        # 我日你妈，torch简写th的是吧
                        with th.no_grad():
                            out = self.p_sample(
                                model,
                                image_after_step,
                                # 接受上一步的噪声图来的
                                t,
                                clip_denoised=clip_denoised,
                                denoised_fn=denoised_fn,
                                cond_fn=cond_fn,
                                model_kwargs=model_kwargs,
                            )
                            # 将当前的t和image_t得到out，t不断变化

                            if postprocess_fn is not None:
                                out = postprocess_fn(out, t)
                            # p_sample + 后处理
                            # 得到的out是根据当前的噪声图来预测的Xt-1

                            yield out
                            # yield就够了 传回参数
                            flag = out["flag"]
                            image_after_step = out["sample"]

                        image_before_step = image_after_step.clone()
                        if r != 9:
                            # 如果不是第10步 貌似用了之后用了前向噪声的的方差来修正
                            image_after_step = self.undo(
                                image_before_step, image_after_step,
                                est_x_0=out['pred_xstart'], t=t - 1, debug=False)
                            # 这传进去的两个参数一样。。。
                        if flag:
                            break
                    if flag:
                        break
                    img = image_after_step  # .clone()
                else:
                    t = th.tensor([i] * shape[0], device=device)
                    bag_image_batch = th.tile(bag_image, dims=(batch_size, 1, 1, 1))
                    img = th.randn(*shape, device=device)
                    true_bag_img = self.q_sample(
                        x_start=bag_image_batch,
                        t=th.tensor([i] * shape[0], device=device),
                        noise=img,
                    )
                    if i / (self.num_timesteps - skip_timesteps - 1) < 0.15:
                        model_kwargs["y"] = th.Tensor([0])
                    else:
                        model_kwargs["y"] = th.Tensor([100])
                    if randomize_class and "y" in model_kwargs:
                        model_kwargs["y"] = th.randint(
                            low=0,
                            high=model.num_classes,
                            size=model_kwargs["y"].shape,
                            device=model_kwargs["y"].device,
                        )
                    with th.no_grad():
                        out = self.p_sample(
                            model,
                            img,
                            t,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            cond_fn=cond_fn,
                            model_kwargs=model_kwargs,
                        )
                        if postprocess_fn is not None:
                            out = postprocess_fn(out, t)

                        out['sample'] = out['sample'][mask] + true_bag_img[1-mask]

                        yield out
                        flag = out["flag"]
                        if flag:
                            break
                        img = out["sample"]

            if i == 0:
                break
                # print(i)
            # if i




    def h_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None,
    ):
        # 从xt去采样出来xt-1
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # out的输出，有这一步出来的均值，方差，预测的x0等等
        # 在推理的时候，会不断地调用p_sample函数来得到x0
        # 用model得出来xt-1均值和方差
        noise = th.randn_like(x)
        # 采样得到去噪的噪声
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        flag = None
        if cond_fn is not None:
            # classifier的时候
            # 需要对mean进行修正 用classifier进行修正
            # 这里就是需要用到的是cond_fn给过来的梯度
            out["mean"], flag = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # 0.5是为了得到标准差
        # 有了均值和方差就可以采样出来xt
        # 标准差 = 0.5乘以一个倍数方差再指数运算
        # 因为0.5是用来开根号的
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        # sample 即为下次的样本，完成一次去噪
        return {"sample": sample, "pred_xstart": out["pred_xstart"],"flag":flag}

    def h_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
        ):
            final = sample
        return final["sample"]
    def h_sample_loop_progressive(
        self,
        model,
        shape,
        mask,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        bag_image=None,
        use_mask=False,
        iterratio=None,
        dual_iter=None,
        mixup=None,
        postprocess_fn=None,
        randomize_class=False,
        classi=None,
        vit_func=None,
        find_init = False
    ):
        # 去推理的一个过程
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        # 生成T时刻（最后一个时刻）一个标准的噪声
        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)
        # 如果不输入init_image则代表让我们进行随机采样

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]
        # 得到一个time_step的时刻，然后将这个range进行逆序，从T..到0

        batch_size = shape[0]
        init_image_batch = th.tile(init_image, dims=(batch_size, 1, 1, 1))
        # bag_image_batch_0 = th.tile(bag_image, dims=(batch_size, 1, 1, 1))
        
        img = self.q_sample(x_start=init_image_batch, t=th.tensor(indices[0], dtype=th.long, device=device), noise=img,)
        # 输入加的噪声XT，第一个时间戳，和原始图片
        # 得到的img是加噪后的图片
        """
        pre_image = img[0].add(1).div(2).clamp(0, 1)
        from torchvision.transforms import functional as TF
        pre_image_pil = TF.to_pil_image(pre_image)
        pre_image_pil.save('outputs/onlyclip/x0.png')
        n_add = th.randn(*shape, device=device)
        gai_img = self.q_sample(
            x_start=bag_image_batch_0,
            t=th.tensor(indices[-20], dtype=th.long, device=device),
            noise=n_add,
        )
        gai = gai_img[0].add(1).div(2).clamp(0, 1)
        e_image_pil = TF.to_pil_image(gai)
        e_image_pil.save('outputs/onlyclip/yt.png')
        """

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        flag = False
        while True:
            if flag:
                indices = list(range(self.num_timesteps - skip_timesteps))[::-1]
                indices = tqdm(indices)
            image_after_step = img
            for i in indices:
                # 对这个时间进行遍历
                if flag:
                    # 除了步骤XT都进来
                    img = th.randn(*shape, device=device)
                    # 生成随机噪声
                    img = self.q_sample(
                            x_start=init_image_batch,
                            t=th.tensor([i] * shape[0], device=device),
                            noise=img,
                        )
                    image_after_step = img
                    # 生成这个timestep的前向加噪图（从后向前）
                if i == self.num_timesteps-skip_timesteps-1:
                    # XT时候从这里
                    for r in range(10):
                        t = th.tensor([i] * shape[0], device=device)

                        # 循环10个timestep
                        if randomize_class and "y" in model_kwargs:
                            model_kwargs["y"] = th.Tensor([636]).long().to(device)
                        model_kwargs["y"] = th.Tensor([636]).long().to(device)
                        # 如果是condition的，并且没有给condition的classlabel
                        # 那么我就自己随机生成一个
                        with th.no_grad():
                            out = self.p_sample(
                                model,
                                image_after_step,
                                # 接受上一步的噪声图来的
                                t,
                                clip_denoised=clip_denoised,
                                denoised_fn=denoised_fn,
                                cond_fn=cond_fn,
                                model_kwargs=model_kwargs,
                            )
                            # 将当前的t和image_t得到out，t不断变化

                            if postprocess_fn is not None:
                                out = postprocess_fn(out, t)
                            # p_sample + 后处理
                            # 得到的out是根据当前的噪声图来预测的Xt-1
                            
                            yield out
                            # yield就够了 传回参数
                            flag = out["flag"] 
                            image_after_step = out["sample"]
                            
                        image_before_step = image_after_step.clone()
                        if r!= 9:
                            # 如果不是第10步 貌似用了之后用了前向噪声的的方差来修正
                            image_after_step = self.undo(
                            image_before_step, image_after_step,
                            est_x_0=out['pred_xstart'], t=t-1, debug=False)
                            # 这传进去的两个参数一样。。。
                        if flag:
                            break
                    if flag:
                        break
                    img = image_after_step#.clone()
                else:
                    t = th.tensor([i] * shape[0], device=device)
                    print(t)

                    bag_image_batch = th.tile(bag_image, dims=(batch_size, 1, 1, 1))
                    ref_img = self.q_sample(
                        x_start=bag_image_batch,
                        t=th.tensor([i] * shape[0], device=device),
                        noise=None,
                    )
                    """
                    
                    fish_img = self.q_sample(
                        x_start=init_image_batch,
                        t=th.tensor([i] * shape[0],device=device),
                        noise=None,
                    )
                    """
                    if randomize_class and "y" in model_kwargs:
                        model_kwargs["y"] =  th.Tensor([636]).long().to(device)
                    model_kwargs["y"] = th.Tensor([636]).long().to(device)
                    with th.no_grad():
                        out = self.p_sample(
                            model,
                            img,
                            t,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            cond_fn=cond_fn,
                            model_kwargs=model_kwargs,
                        )
                        if postprocess_fn is not None:
                            out = postprocess_fn(out, t)

                        yield out
                        flag = out["flag"] 
                        if flag:
                            break
                        img = out["sample"]

                        cls_loss = classi(img, t)
                        x_judge = out["pred_xstart"]
                        cls_j = classi(x_judge,th.Tensor([119]).to(device))
                        vit_loss, vit_loss_val = vit_func(x_judge, bag_image, bag_image, use_dir=False,
                                                          frac_cont=1.0, target=None,
                                                          target_ca=0,mask=None)
                        print(f"the origin cls {cls_loss[0][636]}")
                        print(f"gaijin de loss z{cls_j[0][636]}")
                        print("iopiopiop")
                        print(f"structure loss{vit_loss}")
                        if use_mask and i < (1 - iterratio) * (self.num_timesteps-skip_timesteps-1) and i > (1 - iterratio) * (self.num_timesteps-skip_timesteps-1) -dual_iter :
                            img = img * mask + ref_img * mask.logical_not() * mixup + img * mask.logical_not() * (1-mixup)


            if i==0:
                break
                    # print(i)
            # if i

    def p_sample(
            self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None,
    ):
        # 从xt去采样出来xt-1
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # out的输出，有这一步出来的均值，方差，预测的x0等等
        # 在推理的时候，会不断地调用p_sample函数来得到x0
        # 用model得出来xt-1均值和方差
        noise = th.randn_like(x)
        # 采样得到去噪的噪声
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        flag = None
        if cond_fn is not None:
            # classifier的时候
            # 需要对mean进行修正 用classifier进行修正
            # 这里就是需要用到的是cond_fn给过来的梯度
            out["mean"], flag = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # 0.5是为了得到标准差
        # 有了均值和方差就可以采样出来xt
        # 标准差 = 0.5乘以一个倍数方差再指数运算
        # 因为0.5是用来开根号的
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        # sample 即为下次的样本，完成一次去噪
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "flag": flag}

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            skip_timesteps=0,
            init_image=None,
            randomize_class=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                skip_timesteps=skip_timesteps,
                init_image=init_image,
                randomize_class=randomize_class,
        ):
            final = sample
        return final["sample"]

    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        beta = _extract_into_tensor(self.betas, t, img_out.shape)

        img_in_est = th.sqrt(1 - beta) * img_out + \
                     th.sqrt(beta) * th.randn_like(img_out)

        return img_in_est

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            skip_timesteps=0,
            init_image=None,
            postprocess_fn=None,
            randomize_class=False,
            find_init=False
    ):
        # 去推理的一个过程
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        # 生成T时刻（最后一个时刻）一个标准的噪声
        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)
        # 如果不输入init_image则代表让我们进行随机采样

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]
        # 得到一个time_step的时刻，然后将这个range进行逆序，从T..到0

        batch_size = shape[0]
        init_image_batch = th.tile(init_image, dims=(batch_size, 1, 1, 1))

        img = self.q_sample(x_start=init_image_batch, t=th.tensor(indices[0], dtype=th.long, device=device),
                            noise=img, )
        # 输入加的噪声XT，第一个时间戳，和原始图片
        # 得到的img是加噪后的图片
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        flag = False
        while True:
            if flag:
                indices = list(range(self.num_timesteps - skip_timesteps))[::-1]
                indices = tqdm(indices)
            image_after_step = img
            for i in indices:

                # 对这个时间进行遍历
                if flag:
                    # 除了步骤XT都进来
                    img = th.randn(*shape, device=device)
                    # 生成随机噪声
                    img = self.q_sample(
                        x_start=init_image_batch,
                        t=th.tensor([i] * shape[0], device=device),
                        noise=img,
                    )
                    image_after_step = img
                    # 生成这个timestep的前向加噪图（从后向前）
                if i == self.num_timesteps - skip_timesteps - 1:
                    # XT时候从这里
                    for r in range(10):
                        t = th.tensor([i] * shape[0], device=device)

                        # 循环10个timestep
                        if randomize_class and "y" in model_kwargs:
                            model_kwargs["y"] = th.randint(
                                low=0,
                                high=model.num_classes,
                                size=model_kwargs["y"].shape,
                                device=model_kwargs["y"].device,
                            )
                        # 如果是condition的，并且没有给condition的classlabel
                        # 那么我就自己随机生成一个
                        # 我日你妈，torch简写th的是吧
                        with th.no_grad():
                            out = self.p_sample(
                                model,
                                image_after_step,
                                # 接受上一步的噪声图来的
                                t,
                                clip_denoised=clip_denoised,
                                denoised_fn=denoised_fn,
                                cond_fn=cond_fn,
                                model_kwargs=model_kwargs,
                            )
                            # 将当前的t和image_t得到out，t不断变化

                            if postprocess_fn is not None:
                                out = postprocess_fn(out, t)
                            # p_sample + 后处理
                            # 得到的out是根据当前的噪声图来预测的Xt-1

                            yield out
                            # yield就够了 传回参数
                            flag = out["flag"]
                            image_after_step = out["sample"]

                        image_before_step = image_after_step.clone()
                        if r != 9:
                            # 如果不是第10步 貌似用了之后用了前向噪声的的方差来修正
                            image_after_step = self.undo(
                                image_before_step, image_after_step,
                                est_x_0=out['pred_xstart'], t=t - 1, debug=False)
                            # 这传进去的两个参数一样。。。
                        if flag:
                            break
                    if flag:
                        break
                    img = image_after_step  # .clone()
                else:
                    t = th.tensor([i] * shape[0], device=device)

                    if randomize_class and "y" in model_kwargs:
                        model_kwargs["y"] = th.randint(
                            low=0,
                            high=model.num_classes,
                            size=model_kwargs["y"].shape,
                            device=model_kwargs["y"].device,
                        )
                    with th.no_grad():
                        out = self.p_sample(
                            model,
                            img,
                            t,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            cond_fn=cond_fn,
                            model_kwargs=model_kwargs,
                        )
                        if postprocess_fn is not None:
                            out = postprocess_fn(out, t)

                        yield out
                        flag = out["flag"]
                        if flag:
                            break
                        img = out["sample"]

            if i == 0:
                break
                # print(i)
            # if i
    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        # print("ddim")
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out,flag = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "flag":flag}

    def ddim_reverse_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        postprocess_fn=None,
        randomize_class=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        # print("ddim")
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]
        
        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            batch_size = shape[0]
            init_image_batch = th.tile(init_image, dims=(batch_size, 1, 1, 1))
            img = self.q_sample(init_image_batch, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        # indices = tqdm(indices)
        flag = False
        while True:
            if flag:
                indices = list(range(self.num_timesteps - skip_timesteps))[::-1]
                # from tqdm.auto import tqdm
                indices = tqdm(indices)
            for i in indices:
                # t = th.tensor([i] * shape[0], device=device)
                if flag:
                    img = th.randn(*shape, device=device)
                    # my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[i]
                    my_t=th.tensor([i] * shape[0], device=device)
                    # my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[i]
                    batch_size = shape[0]
                    init_image_batch = th.tile(init_image, dims=(batch_size, 1, 1, 1))
                    img = self.q_sample(init_image_batch, my_t, img)

                # t = th.tensor([i] * shape[0], device=device)
                # t = th.tensor([i] * shape[0], device=device).long()
                # print(t)
                t = th.tensor([i] * shape[0], device=device)
                if randomize_class and "y" in model_kwargs:
                    model_kwargs["y"] = th.randint(
                        low=0,
                        high=model.num_classes,
                        size=model_kwargs["y"].shape,
                        device=model_kwargs["y"].device,
                    )
                with th.no_grad():
                    # print(t)
                    out = self.ddim_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                    )

                    if postprocess_fn is not None:
                        out = postprocess_fn(out, t)

                    yield out
                    img = out["sample"]
                    flag = out["flag"]
                    # print(flag)
                    if flag:
                        # print('break')
                        break
                    # img = out["sample"]
            if i==0:
                break
#     def ddim_sample_loop_progressive(
#         self,
#         model,
#         shape,
#         noise=None,
#         clip_denoised=True,
#         denoised_fn=None,
#         cond_fn=None,
#         model_kwargs=None,
#         device=None,
#         progress=False,
#         eta=0.0,
#         skip_timesteps=0,
#         init_image=None,
#         postprocess_fn=None,
#         randomize_class=False,
#     ):
#         """
#         Use DDIM to sample from the model and yield intermediate samples from
#         each timestep of DDIM.
#         Same usage as p_sample_loop_progressive().
#         """
#         if device is None:
#             device = next(model.parameters()).device
#         assert isinstance(shape, (tuple, list))
#         if noise is not None:
#             img = noise
#         else:
#             img = th.randn(*shape, device=device)

#         if skip_timesteps and init_image is None:
#             init_image = th.zeros_like(img)

#         indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

#         if init_image is not None:
#             my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
#             batch_size = shape[0]
#             init_image_batch = th.tile(init_image, dims=(batch_size, 1, 1, 1))
#             img = self.q_sample(init_image_batch, my_t, img)

#         if progress:
#             # Lazy import so that we don't depend on tqdm.
#             from tqdm.auto import tqdm

#             indices = tqdm(indices)

#         for i in indices:
#             t = th.tensor([i] * shape[0], device=device)
#             if randomize_class and "y" in model_kwargs:
#                 model_kwargs["y"] = th.randint(
#                     low=0,
#                     high=model.num_classes,
#                     size=model_kwargs["y"].shape,
#                     device=model_kwargs["y"].device,
#                 )
#             with th.no_grad():
#                 out = self.ddim_sample(
#                     model,
#                     img,
#                     t,
#                     clip_denoised=clip_denoised,
#                     denoised_fn=denoised_fn,
#                     cond_fn=cond_fn,
#                     model_kwargs=model_kwargs,
#                     eta=eta,
#                 )

#                 if postprocess_fn is not None:
#                     out = postprocess_fn(out, t)

#                 yield out
#                 img = out["sample"]
    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
