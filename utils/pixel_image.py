import torch
from torch.distributions import MultivariateNormal, Categorical
import torch.nn.functional as F


class OnlineGMMUpdater:
    def __init__(self, feature_dim=1, num_components=4, regularization=1e-6):
        self.feature_dim = feature_dim
        self.K = num_components
        self.reg = regularization

        self.weights = torch.ones(self.K, device='cuda') / self.K
        self.means = torch.rand(self.K, self.feature_dim, device='cuda')
        self.covariances = torch.eye(self.feature_dim, device='cuda').unsqueeze(0).repeat(self.K, 1, 1) * self.reg

        self.n_samples_per_component = torch.zeros(self.K, device='cuda')

    def _get_responsibilities(self, pixels):
        log_probs = torch.zeros(pixels.shape[0], self.K, device=pixels.device)
        for k in range(self.K):
            try:
                dist = MultivariateNormal(self.means[k], self.covariances[k])
                log_probs[:, k] = dist.log_prob(pixels)
            except ValueError:
                log_probs[:, k] = -1e10

        log_probs += torch.log(self.weights + 1e-8)  
        responsibilities = torch.exp(log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True))
        return responsibilities

    def update(self, pixels):
        if pixels.shape[0] == 0: return
        pixels = pixels.cuda()
        responsibilities = self._get_responsibilities(pixels)
        R_k = responsibilities.sum(dim=0)

        for k in range(self.K):
            if R_k[k] < 1e-6: continue
            mu_k_batch = torch.sum(responsibilities[:, k].unsqueeze(1) * pixels, dim=0) / R_k[k]
            n_new = self.n_samples_per_component[k] + R_k[k]
            self.means[k] = (self.n_samples_per_component[k] * self.means[k] + R_k[k] * mu_k_batch) / n_new

            diff = pixels - self.means[k] 
            cov_k_batch = torch.matmul((responsibilities[:, k].unsqueeze(1) * diff).T, diff) / R_k[k]
            self.covariances[k] = (self.n_samples_per_component[k] * self.covariances[k] + R_k[k] * cov_k_batch) / n_new
            self.covariances[k] += torch.eye(self.feature_dim, device='cuda') * self.reg

            self.n_samples_per_component[k] = n_new

        total_samples = self.n_samples_per_component.sum()
        if total_samples > 0:
            self.weights = self.n_samples_per_component / total_samples

    def sample(self, num_samples):
        if self.K == 0:
            return torch.zeros(num_samples, self.feature_dim, device='cuda')

        k_prototypes = self.means
        num_repeats_full = num_samples // self.K

        samples_list = []
        if num_repeats_full > 0:
            samples_list.append(k_prototypes.repeat(num_repeats_full, 1))
        remaining = num_samples % self.K
        if remaining > 0:
            samples_list.append(k_prototypes[:remaining])
        if not samples_list:
            return k_prototypes[:num_samples]
        final_samples = torch.cat(samples_list, dim=0)
        final_samples = final_samples[torch.randperm(num_samples)]

        return final_samples


class MultiClassGaussianUpdater:
    def __init__(self, feature_dim=1, num_classes=5, regularization=1e-6, K=4):  # 增加一个K参数
        self.num_classes = num_classes
        self.updaters = {
            cls: OnlineGMMUpdater(feature_dim, num_components=K, regularization=regularization)
            for cls in range(num_classes)
        }

    def update(self, image, mask):
        B, C, H, W = image.shape
        pixels_flat = image.permute(0, 2, 3, 1).reshape(-1, C)
        mask_flat = mask.permute(1, 0, 2, 3).reshape(self.num_classes, -1)
        for cls in range(self.num_classes):
            category_mask_flat = mask_flat[cls].bool()
            if category_mask_flat.sum() > 0:
                category_pixels = pixels_flat[category_mask_flat]
                if category_pixels.shape[0] > 8192:
                    indices = torch.randperm(category_pixels.shape[0])[:8192]
                    category_pixels = category_pixels[indices]
                self.updaters[cls].update(category_pixels)

    def sample(self, num_samples_per_class):
        samples = []
        for cls in range(self.num_classes):
            samples.append(self.updaters[cls].sample(num_samples_per_class))
        return torch.stack(samples, dim=0)


def generate_fusion_image_batch_smooth(sourcelike_batch, targetlike_batch, mask_batch, source_updater, target_updater):
    device = sourcelike_batch.device
    B, C, H, W = sourcelike_batch.shape

    target_fusion_images = sourcelike_batch.clone()
    source_fusion_images = targetlike_batch.clone()

    source_samples = sourcelike_batch.clone()
    target_samples = targetlike_batch.clone()

    batch_labels = mask_batch.argmax(dim=1).to(device)

    for b in range(B):
        unique_categories, counts = torch.unique(batch_labels[b], return_counts=True)
        area_threshold = 512
        unique_categories = unique_categories[counts >= area_threshold]
     
        unique_categories = unique_categories[unique_categories != 0]

        if unique_categories.numel() <= 1:
            continue

        random_category = unique_categories[torch.randint(0, len(unique_categories), (1,))].item()

        mask = (batch_labels[b] == random_category)

        t_gen_sample = target_updater.updaters[random_category].sample(H * W).T.reshape(1, H, W)
        t_gen_sample = smooth_image(t_gen_sample).cuda()

        target_fusion_images[b][:, mask] = t_gen_sample[:, mask]
        if C > 1 and t_gen_sample.shape[0] == 1:
            target_samples[b] = t_gen_sample.repeat(C, 1, 1)
        else:
            target_samples[b] = t_gen_sample

        target_samples[b][:, ~mask] = 0

        s_gen_sample = source_updater.updaters[random_category].sample(H * W).T.reshape(1, H, W)
        s_gen_sample = smooth_image(s_gen_sample).cuda()

        source_fusion_images[b][:, mask] = s_gen_sample[:, mask]

        if C > 1 and s_gen_sample.shape[0] == 1:
            source_samples[b] = s_gen_sample.repeat(C, 1, 1)
        else:
            source_samples[b] = s_gen_sample

        source_samples[b][:, ~mask] = 0

    source_fusion_images = torch.clamp(source_fusion_images, 0, 1)
    target_fusion_images = torch.clamp(target_fusion_images, 0, 1)

    return source_fusion_images, target_fusion_images, source_samples, target_samples


def smooth_image(image, kernel_size=5, sigma=1.0):
    C, H, W = image.shape
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss_kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_kernel_1d /= gauss_kernel_1d.sum()
    gauss_kernel_2d = gauss_kernel_1d[:, None] @ gauss_kernel_1d[None, :]
    gauss_kernel_2d = gauss_kernel_2d.to(image.device).unsqueeze(0).unsqueeze(0)

    smoothed_image = F.conv2d(image.unsqueeze(0), gauss_kernel_2d, padding=kernel_size // 2, groups=1)
    return smoothed_image.squeeze(0)
