""""
    The below two classes of this file's PixelCNN code are originally adapted from
    https://github.com/pbloem/pixel-models and have been modified to fit the rest of our code.

    MIT License

    Copyright (c) 2018 Peter Bloem

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import torch
import torch.nn as nn


class LMaskedConv2d(nn.Module):
    """
    Masked convolution, with location dependent conditional.
    The conditional must be an 'image' tensor (BCHW) with the same resolution as the instance
    (no of channels can be different)
    """

    def __init__(self, conditional_channels, channels, colors=3, self_connection=False, res_connection=True,
                 hv_connection=True, gates=True, k=7, padding=3):

        super().__init__()

        assert (k // 2) * 2 == k - 1  # only odd numbers accepted

        self.gates = gates
        self.res_connection = res_connection
        self.hv_connection = hv_connection

        f = 2 if self.gates else 1

        self.vertical = nn.Conv2d(channels, channels * f, kernel_size=k, padding=padding, bias=False)
        self.horizontal = nn.Conv2d(channels, channels * f, kernel_size=(1, k), padding=(0, padding), bias=False)
        self.tohori = nn.Conv2d(channels * f, channels * f, kernel_size=1, padding=0, bias=False, groups=colors)
        self.tores = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False, groups=colors)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        self.vmask[:, :, k // 2:, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2:] = 0

        # Add connections to "previous" colors (G is allowed to see R, and B is allowed to see R and G)
        m = k // 2  # index of the middle of the convolution
        pc = channels // colors  # channels per color

        for c in range(0, colors):
            f, t = c * pc, (c + 1) * pc

            if f > 0:
                self.hmask[f:t, :f, 0, m] = 1
                self.hmask[f + channels:t + channels, :f, 0, m] = 1

            # Connections to "current" colors (but not "future colors", R is not allowed to see G and B)
            if self_connection:
                self.hmask[f:t, :f + pc, 0, m] = 1
                self.hmask[f + channels:t + channels, :f + pc, 0, m] = 1

        # The conditional weights
        self.vhf = nn.Conv2d(conditional_channels, channels, 1)
        self.vhg = nn.Conv2d(conditional_channels, channels, 1)
        self.vvf = nn.Conv2d(conditional_channels, channels, 1)
        self.vvg = nn.Conv2d(conditional_channels, channels, 1)

    def forward(self, vxin, hxin, h):
        self.vertical.weight.data *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx = self.vertical.forward(vxin)
        hx = self.horizontal.forward(hxin)

        if self.hv_connection:
            hx = hx + self.tohori(vx)

        if self.gates:
            vx = self.gate(vx, h, (self.vvf, self.vvg))
            hx = self.gate(hx, h, (self.vhf, self.vhg))

        if self.res_connection:
            hx = hxin + self.tores(hx)

        return vx, hx

    def gate(self, x, cond, weights):
        """
        Takes a batch x channels x rest... tensor and applies an LTSM-style gate activation.
        - The top half of the channels are fed through a tanh activation, functioning as the activated neurons
        - The bottom half are fed through a sigmoid, functioning as a mask
        - The two are element-wise multiplied, and the result is returned.
        Conditional and weights are used to compute a bias based on the conditional element.
        """

        # compute conditional term
        vf, vg = weights
        tan_bias = vf(cond)
        sig_bias = vg(cond)

        # compute convolution term
        half = x.size(1) // 2

        top = x[:, :half]
        bottom = x[:, half:]

        # apply gate and return
        return torch.tanh(top + tan_bias) * torch.sigmoid(bottom + sig_bias)


class PixelCNN(nn.Module):
    """
    Gated PixelCNN decoder with flexible amount of layers, kernel sizes, conditional channels and input dimensionality.
    """
    def __init__(self, device, num_colors, conditional_channels, channels, num_layers, k=7, padding=3):
        super().__init__()

        self.device = device
        self.num_colors = num_colors

        self.conv1 = nn.Conv2d(self.num_colors, channels, 1, groups=self.num_colors)

        self.gated_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gated_layers.append(
                LMaskedConv2d(conditional_channels, channels, colors=self.num_colors, self_connection=i > 0,
                              res_connection=i > 0, gates=True, hv_connection=True, k=k, padding=padding)
            )

        self.conv2 = nn.Conv2d(channels, 256 * self.num_colors, 1, groups=self.num_colors)

    def generate(self, x):
        samples = torch.zeros((x.size(0), self.num_colors, x.size(2), x.size(3))).to(self.device)
        for h in range(x.size(2)):
            for w in range(x.size(3)):
                for c in range(self.num_colors):
                    result = self.forward(samples, x)
                    probs = torch.softmax(result[:, :, c, h, w], dim=1).data

                    pixel_sample = torch.multinomial(probs, 1).float() / 255.
                    samples[:, c, h, w] = pixel_sample.squeeze()

        return samples

    def forward(self, x, cond):
        b, c, h, w = x.size()

        x = self.conv1(x)
        xv, xh = x, x

        for layer in self.gated_layers:
            xv, xh = layer(xv, xh, cond)

        x = self.conv2(xh)

        return x.view(b, c, 256, h, w).transpose(1, 2)
