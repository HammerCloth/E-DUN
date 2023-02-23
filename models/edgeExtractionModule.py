import numpy as np
import torch
import torch.nn as nn
from scipy import signal as signal


class CandyNet(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=False):
        super(CandyNet, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = signal.gaussian(filter_size, std=1.0).reshape([1, filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(1, 1, kernel_size=(1, filter_size), padding=(0, filter_size // 2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        self.gaussian_filter_vertical = nn.Conv2d(1, 1, kernel_size=(filter_size, 1), padding=(filter_size // 2, 0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(1, 1, kernel_size=sobel_filter.shape,
                                                 padding=sobel_filter.shape[0] // 2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        self.sobel_filter_vertical = nn.Conv2d(1, 1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0] // 2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([[0, 0, 0],
                             [0, 1, -1],
                             [0, 0, 0]])

        filter_45 = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, -1]])

        filter_90 = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, -1, 0]])

        filter_135 = np.array([[0, 0, 0],
                               [0, 1, 0],
                               [-1, 0, 0]])

        filter_180 = np.array([[0, 0, 0],
                               [-1, 1, 0],
                               [0, 0, 0]])

        filter_225 = np.array([[-1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]])

        filter_270 = np.array([[0, -1, 0],
                               [0, 1, 0],
                               [0, 0, 0]])

        filter_315 = np.array([[0, 0, -1],
                               [0, 1, 0],
                               [0, 0, 0]])

        all_filters = np.stack(
            [filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(1, 8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img):  # (batch,channel,height, width)

        batch = img.shape[0]
        img_r = img[:, 0:1]  # batch,1,height, width
        img_g = img[:, 1:2]  # batch,1,height, width
        img_b = img[:, 2:3]  # batch,1,height, width

        blur_horizontal = self.gaussian_filter_horizontal(img_r)  # batch,1,height,width
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)  # batch,1,height,width
        blur_horizontal = self.gaussian_filter_horizontal(img_g)  # batch,1,height,width
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)  # batch,1,height,width
        blur_horizontal = self.gaussian_filter_horizontal(img_b)  # batch,1,height,width
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)  # batch,1,height,width

        blurred_img = torch.stack([blurred_img_r, blurred_img_g, blurred_img_b], dim=1)  # batch,1,height,width
        blurred_img = torch.stack([torch.squeeze(blurred_img)])  # batch,1,height,width

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)  # batch,1,height,width
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)  # batch,1,height,width
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)  # batch,1,height,width
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)  # batch,1,height,width
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)  # batch,1,height,width
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)  # batch,1,height,width

        # COMPUTE THICK EDGES
        grad_mag = torch.sqrt(grad_x_r ** 2 + grad_y_r ** 2)  # batch,1,height,width
        grad_mag += torch.sqrt(grad_x_g ** 2 + grad_y_g ** 2)  # batch,1,height,width
        grad_mag += torch.sqrt(grad_x_b ** 2 + grad_y_b ** 2)  # batch,1,height,width
        grad_orientation = (  # batch,1,height,width
                torch.atan2(grad_y_r + grad_y_g + grad_y_b, grad_x_r + grad_x_g + grad_x_b) * (180.0 / 3.14159))
        grad_orientation += 180.0  # batch,1,height,width
        grad_orientation = torch.round(grad_orientation / 45.0) * 45.0  # batch,1,height,width

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)  # batch,8,height,width
        inidices_positive = (grad_orientation / 45) % 8  # batch,1,height,width
        inidices_negative = ((grad_orientation / 45) + 4) % 8  # batch,1,height,width

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width

        pixel_range = torch.FloatTensor([range(pixel_count)])  # batch,pixel_range
        if self.use_cuda:
            pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        indices = (  # batch,pixel_range
                inidices_positive.view(
                    inidices_positive.shape[0],
                    pixel_count).data * pixel_count + pixel_range)

        channel_select_filtered_positive = torch.ones(batch, 1, height, width)  # batch, 1, height, width
        for i in range(batch):
            channel_select_filtered_positive_temp = all_filtered[i].view(-1)[indices[i].long()].view(1, height, width)
            channel_select_filtered_positive[i] = channel_select_filtered_positive_temp

        indices = (  # batch,pixel_range
                inidices_negative.view(
                    inidices_negative.shape[0],
                    pixel_count).data * pixel_count + pixel_range)

        channel_select_filtered_negative = torch.ones(batch, 1, height, width)  # batch, 1, height, width
        for i in range(batch):
            channel_select_filtered_negative_temp = all_filtered[i].view(-1)[indices[i].long()].view(1, height, width)
            channel_select_filtered_negative[i] = channel_select_filtered_negative_temp

        channel_select_filtered = torch.stack(  # batch, 2, height, width
            [channel_select_filtered_positive, channel_select_filtered_negative], dim=1)

        is_max = channel_select_filtered.min(dim=1)[0] > 0.0

        thin_edges = grad_mag.clone()
        thin_edges[is_max == 0] = 0.0

        # THRESHOLD

        thresholded = thin_edges.clone()
        thresholded[thin_edges < self.threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag < self.threshold] = 0.0

        # assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        return thresholded


if __name__ == '__main__':
    CandyNet()
