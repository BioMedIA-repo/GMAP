import os

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt, cm
from monai.visualize import GradCAM
from torch.nn import init
# from torchcam.methods import SmoothGradCAMpp
# from torchcam.utils import overlay_mask
from torchsummary import summary
import torchvision.utils as vutils
from PIL import Image
from torchvision.transforms.functional import to_pil_image


class Encoder(nn.Module):
    def __init__(self, in_channels, filters, dropout_rate):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.dropout_rate = dropout_rate
        factor = 2
        self.conv1 = UnetConv2D(self.in_channels, self.filters[0], self.dropout_rate[0])
        self.conv2 = Down(self.filters[0], self.filters[1], self.dropout_rate[1])
        self.conv3 = Down(self.filters[1], self.filters[2], self.dropout_rate[2])
        self.conv4 = Down(self.filters[2], self.filters[3], self.dropout_rate[3])
        self.center = Down(self.filters[3], self.filters[4] // factor, self.dropout_rate[4])

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        center = self.center(conv4)
        return conv1, conv2, conv3, conv4, center


class Decoder(nn.Module):
    def __init__(self, out_channels, filters, dropout_rate):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.filters = filters
        self.dropout_rate = dropout_rate

        factor = 2

        self.up_concat4 = Up(self.filters[4], self.filters[3] // factor, self.dropout_rate[0])
        self.up_concat3 = Up(self.filters[3], self.filters[2] // factor, self.dropout_rate[1])
        self.up_concat2 = Up(self.filters[2], self.filters[1] // factor, self.dropout_rate[2])
        self.up_concat1 = Up(self.filters[1], self.filters[0], self.dropout_rate[3])

        self.final = OutConv(self.filters[0], self.out_channels)

    def forward(self, conv1, conv2, conv3, conv4, center):
        dec4 = self.up_concat4(center, conv4)
        dec3 = self.up_concat3(dec4, conv3)
        dec2 = self.up_concat2(dec3, conv2)
        dec1 = self.up_concat1(dec2, conv1)
        final = self.final(dec1)
        soft_final = F.softmax(final, dim=1)
        return soft_final


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, encoder_dropout_rate=None, decoder_dropout_rate=None,
                 use_projection=True):
        super(UNet, self).__init__()
        if encoder_dropout_rate is None:
            encoder_dropout_rate = [0, 0, 0, 0, 0]
        if decoder_dropout_rate is None:
            decoder_dropout_rate = [0, 0, 0, 0]
        self.filters = [16, 32, 64, 128, 256]
        self.encoder_dropout_rate = encoder_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate

        self.use_projection = use_projection
        if self.use_projection:
            proj_out_dim = 128
            self.projection = ProjectionLayer(output_dim=proj_out_dim)  # 只指定输出维度
        self.encoder = Encoder(in_channels, self.filters, self.encoder_dropout_rate)
        self.decoder = Decoder(out_channels, self.filters, self.decoder_dropout_rate)

    def forward(self, inputs):
        conv1, conv2, conv3, conv4, center = self.encoder(inputs)
        soft_final = self.decoder(conv1, conv2, conv3, conv4, center)
        # return conv1, conv2, conv3, conv4, center
        return soft_final

    # def set_trainable_layers(self, layer_names):
    #     # 首先冻结所有编码器层
    #     for name, param in self.encoder.named_parameters():
    #         param.requires_grad = False
    #
    #     # 解冻指定的层
    #     for name, param in self.named_parameters():
    #         if any(layer in name for layer in layer_names):
    #             param.requires_grad = True


class ProjectionLayer(nn.Module):
    def __init__(self, output_dim=128):
        super(ProjectionLayer, self).__init__()
        self.output_dim = output_dim
        self.mid_dim = 512
        self.linear = nn.Linear(32768, self.mid_dim)
        self.proj = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.mid_dim, output_dim)
        )

    def forward(self, x):
        # x 形状: [B, C, H, W]
        x = torch.flatten(x, start_dim=1).cuda()  # 变形为 [B, C*H*W]
        x = self.linear(x)
        x = self.proj(x)
        x = F.normalize(x, p=2, dim=1)  # L2 归一化
        return x


class UnetConv2D(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0.):
        super(UnetConv2D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)
        return outputs


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            UnetConv2D(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = UnetConv2D(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# def save_center_feature_cam(model, input_tensor, save_dir, i):
#     """
#     保存特定层的特征类别激活图（CAM）。
#
#     Args:
#         model: PyTorch 模型。
#         input_tensor: 输入张量，形状为 (B, C, H, W)。
#         save_dir: 保存CAM图的目录。
#         i: 想要可视化的层的索引。
#     """
#     # 检查保存目录是否存在，不存在则创建
#     os.makedirs(save_dir, exist_ok=True)
#     model.eval()
#
#     # 指定目标层为 Encoder 的 `center`
#     target_layer = "encoder.center"
#
#     # 初始化 CAM 方法，并指定目标层
#     cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)
#
#     output = model(input_tensor)
#     output = torch.argmax(output, dim=1)
#     pred = output.cpu().numpy()  # 获取预测结果
#
#     # 获取每个类别的 CAM 图
#     for class_idx in range(5):  # 遍历所有类别
#         cam = cam_extractor(class_idx, output)
#
#         # 叠加原图和 CAM 图
#         cam = cam[0].cpu()
#
#         # 将 CAM 图归一化到 [0, 1]
#         cam = (cam - cam.min()) / (cam.max() - cam.min())
#
#         # 调整 CAM 图大小以匹配输入图像尺寸
#         cam_resized = F.interpolate(cam.unsqueeze(0), size=input_tensor.shape[2:], mode="bilinear",
#                                     align_corners=False).squeeze()
#
#         # 转为 NumPy 格式
#         cam_resized_np = cam_resized.cpu().numpy()
#
#         # 将输入图像转换为 PIL 图像格式（用于可视化）
#         input_image = input_tensor[0].permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C)
#         input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())  # 归一化到 [0, 1]
#
#         # 创建叠加图：将原图和 CAM 图融合
#         heatmap = plt.cm.jet(cam_resized_np)[:, :, :3]  # 使用 Jet 颜色映射
#         overlay_image = 0.8 * input_image + 0.5 * heatmap  # 图像叠加，设置透明度
#         overlay_image = (overlay_image - overlay_image.min()) / (overlay_image.max() - overlay_image.min())  # 归一化到 [0, 1]
#
#         # 保存叠加图像
#         save_path = os.path.join(save_dir, f"{i}_{class_idx}_overlay.png")
#         plt.imsave(save_path, overlay_image)
#         # print(f"Saved overlaid CAM for class {class_idx} at {save_path}")
#
#         # 转为 RGB 格式并归一化
#         pred_resized = np.expand_dims(pred[0], axis=-1)  # 转为 (H, W, 1)
#         pred_resized = np.repeat(pred_resized, 3, axis=-1)  # 扩展到 (H, W, 3)
#         pred_resized = pred_resized / pred_resized.max()  # 归一化到 [0, 1]
#
#         input_image = np.repeat(input_image, 3, axis=-1)  # 扩展到 (H, W, 3)
#         concat_image = np.concatenate((input_image, pred_resized, overlay_image), axis=1)
#
#         # 保存拼接后的图像
#         concat_save_path = os.path.join(save_dir, f"{i}_{class_idx}_concat.png")
#         plt.imsave(concat_save_path, concat_image)
#         # print(f"Saved concatenated image for class {class_idx} at {concat_save_path}")



def save_feature_maps_as_images(model, input_tensor, save_dir):
    """
    Save the feature maps from each layer of the model as grid images.

    :param model: The neural network model.
    :param input_tensor: The input tensor to the model.
    :param save_dir: The directory to save the images.
    """
    model.eval()  # Set the model to evaluation mode

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        # Get the features from the model
        conv1, conv2, conv3, conv4, center, dec4, dec3, dec2, dec1, final, _ = model(input_tensor)
        encoder_features = [conv1, conv2, conv3, conv4, center]
        decoder_features = [dec4, dec3, dec2, dec1, final]
        all_features = encoder_features + decoder_features

        for layer_idx, feature in enumerate(all_features):
            num_channels = feature.size(1)
            grid_size = math.ceil(math.sqrt(num_channels))

            # Create an empty grid to hold all feature maps
            grid_height = feature.size(2) * grid_size
            grid_width = feature.size(3) * grid_size
            grid_image = np.zeros((grid_height, grid_width))

            for i in range(num_channels):
                # Get the feature map
                feature_map = feature[0, i].cpu().numpy()

                # Normalize the feature map to [0, 255]
                min_val = feature_map.min()
                max_val = feature_map.max()
                if min_val != max_val:
                    feature_map = (feature_map - min_val) / (max_val - min_val)
                    feature_map = (feature_map * 255).astype(np.uint8)
                else:
                    feature_map = np.zeros_like(feature_map, dtype=np.uint8)

                # Calculate the position in the grid
                row = i // grid_size
                col = i % grid_size

                # Place the feature map in the grid
                grid_image[row * feature.size(2):(row + 1) * feature.size(2),
                col * feature.size(3):(col + 1) * feature.size(3)] = feature_map

                # Save the grid image
            im = Image.fromarray(grid_image)
            im = im.convert("L")  # Convert to grayscale
            im.save(os.path.join(save_dir, f'layer_{layer_idx}_grid.png'))


def save_all_feature_map(model, input_tensor, save_dir, i):
    """
    Visualize the feature maps, interpolate them to 256x256, compute the average,
    and save the final image.

    :param model: The neural network model.
    :param input_tensor: The input tensor to the model.
    :param save_dir: The directory to save the images.
    :param i: Identifier for the current input (used in file naming).
    """
    model.eval()  # Set the model to evaluation mode

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        # Get the features from the model
        conv1, conv2, conv3, conv4, center = model(input_tensor)

        if not isinstance(center, torch.Tensor):
            raise ValueError("Expected final to be a torch.Tensor")

        # Convert input tensor to a numpy array for the original image
        input_image = input_tensor[0][0].cpu().numpy()
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())  # Normalize to [0, 1]
        input_image = (input_image * 255).astype(np.uint8)

        # List to hold resized feature maps
        resized_feature_maps = []

        # Resize each layer's feature map to 256x256 and store
        for feature_map in [conv1, conv2, conv3, conv4, center]:
            feature_map_resized = F.interpolate(
                feature_map,
                size=(256, 256),  # Resize to 256x256
                mode='bilinear',
                align_corners=False
            ).mean(dim=1, keepdim=True)  # Take the mean across channels

            # Normalize the feature map to [0, 1]
            min_val, max_val = feature_map_resized.min(), feature_map_resized.max()
            if min_val != max_val:
                feature_map_resized = (feature_map_resized - min_val) / (max_val - min_val)
            else:
                feature_map_resized = torch.zeros_like(feature_map_resized)

            resized_feature_maps.append(feature_map_resized)

        # Compute the average of all resized feature maps
        avg_feature_map = torch.mean(torch.stack(resized_feature_maps), dim=0)

        # Convert to numpy for visualization
        avg_feature_map = avg_feature_map.squeeze().cpu().numpy()

        # Normalize and convert to [0, 255] range for visualization
        avg_feature_map = (avg_feature_map * 255).astype(np.uint8)

        # Convert feature map to a color map using matplotlib
        colormap = cm.get_cmap('viridis')  # Using 'viridis' colormap
        feature_map_color = colormap(avg_feature_map)  # Apply colormap
        feature_map_color = (feature_map_color[:, :, :3] * 255).astype(
            np.uint8)  # Take RGB channels and convert to uint8

        # Convert feature map to a PIL image
        feature_map_image = Image.fromarray(feature_map_color)

        # Convert original image to PIL image
        input_image_pil = Image.fromarray(input_image)

        # Concatenate original image and feature map image
        combined_image = Image.new('RGB', (input_image_pil.width + feature_map_image.width, input_image_pil.height))
        combined_image.paste(input_image_pil, (0, 0))
        combined_image.paste(feature_map_image.convert('RGB'), (input_image_pil.width, 0))

        # Save the combined image
        save_path = os.path.join(save_dir, f"{i}_combined.png")
        combined_image.save(save_path)
        # print(f"Saved combined image at {save_path}")

def save_one_feature_map(model, input_tensor, save_dir, i):
    """
    Visualize the feature maps and concatenate them with the original image.

    :param model: The neural network model.
    :param input_tensor: The input tensor to the model.
    :param save_dir: The directory to save the images.
    :param i: Identifier for the current input (used in file naming).
    """
    model.eval()  # Set the model to evaluation mode

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        # Get the features from the model
        conv1, conv2, conv3, conv4, center = model(input_tensor)

        if not isinstance(center, torch.Tensor):
            raise ValueError("Expected final to be a torch.Tensor")

        # Convert input tensor to a numpy array for the original image
        input_image = input_tensor[0][0].cpu().numpy()
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())  # Normalize to [0, 1]
        input_image = (input_image * 255).astype(np.uint8)

        # Get the feature map (mean over channels)
        feature_map = center.mean(dim=1, keepdim=True)

        # Normalize the feature map to [0, 1]
        min_val, max_val = feature_map.min(), feature_map.max()
        if min_val != max_val:
            feature_map = (feature_map - min_val) / (max_val - min_val)
        else:
            feature_map = np.zeros_like(feature_map)

        # Resize feature map to match the original image size
        feature_map_resized = F.interpolate(
            feature_map,
            size=(256, 256),  # Use original image size
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        # Convert feature map to [0, 255] range for visualization
        feature_map_resized = (feature_map_resized * 255).astype(np.uint8)

        # Convert feature map to a color map using matplotlib
        colormap = cm.get_cmap('viridis')  # inferno        magma.
        feature_map_color = colormap(feature_map_resized)  # Apply colormap
        feature_map_color = (feature_map_color[:, :, :3] * 255).astype(np.uint8)  # Take RGB channels and convert to uint8

        # Convert feature map to a PIL image
        feature_map_image = Image.fromarray(feature_map_color)

        # Convert original image to PIL image
        input_image_pil = Image.fromarray(input_image)

        # Concatenate original image and feature map image
        combined_image = Image.new('RGB', (input_image_pil.width + feature_map_image.width, input_image_pil.height))
        combined_image.paste(input_image_pil, (0, 0))
        combined_image.paste(feature_map_image.convert('RGB'), (input_image_pil.width, 0))

        # Save the combined image
        save_path = os.path.join(save_dir, f"{i}_combined.png")
        combined_image.save(save_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=5).to(device)
    summary(model, input_size=(1, 256, 256))


if __name__ == "__main__":
    main()
