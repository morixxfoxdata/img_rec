import torch
import torch.nn as nn

class GIDC28(nn.Module):
    def __init__(self, kernel_size=5, name="GI", select="both"):
        """
        kernel_size: 畳み込み層・転置畳み込み層で使用するカーネルサイズ。
        "same" パディングの場合、内部で (kernel_size-1)//2 を計算
        """
        super(GIDC28, self).__init__()
        self.kernel_size = kernel_size
        self.model_name = name if name else self.__class__.__name__
        # エンコーダ部分
        self.down1_1 = self.down_sample(
            1, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.down1_2 = self.down_sample(
            16, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.down1_3 = self.down_sample(
            16, 32, kernel_size=kernel_size, stride=2, padding="same"
        )

        self.down2_1 = self.down_sample(
            32, 32, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.down2_2 = self.down_sample(
            32, 32, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.down2_3 = self.down_sample(
            32, 64, kernel_size=kernel_size, stride=2, padding="same"
        )

        self.bottom = nn.Sequential(
            nn.Conv2d(
                64,
                64,
                kernel_size=kernel_size,
                stride=1,
                padding=self._get_padding(kernel_size, "same"),
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        # デコーダ部分
        self.up1_1 = self.up_sample(
            64, 32, kernel_size=kernel_size, stride=2, padding="same"
        )
        self.up1_2 = self.down_sample(
            64, 32, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.up1_3 = self.down_sample(
            32, 32, kernel_size=kernel_size, stride=1, padding="same"
        )

        self.up2_1 = self.up_sample(
            32, 16, kernel_size=kernel_size, stride=2, padding="same"
        )
        self.up2_2 = self.down_sample(
            32, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.up2_3 = self.down_sample(
            16, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        if select == "both":
            self.final = nn.Sequential(
                nn.Conv2d(
                    16,
                    1,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=self._get_padding(kernel_size, "same"),
                ),
                nn.BatchNorm2d(1),
                nn.Tanh(),
            )
        else:            
            self.final = nn.Sequential(
                nn.Conv2d(
                    16,
                    1,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=self._get_padding(kernel_size, "same"),
                ),
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            )

    def _get_padding(self, kernel_size, padding):
        """
        padding引数が "same" の場合、(kernel_size-1)//2 を返す。
        それ以外の場合はそのままの値を返す。
        """
        if padding == "same":
            return (kernel_size - 1) // 2
        else:
            return padding

    def down_sample(self, in_channels, out_channels, kernel_size, stride, padding):
        pad = self._get_padding(kernel_size, padding)
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def up_sample(self, in_channels, out_channels, kernel_size, stride, padding):
        pad = self._get_padding(kernel_size, padding)
        # 出力サイズを (input_size * stride) にするための output_padding を自動計算
        output_padding = stride + 2 * pad - kernel_size
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                output_padding=output_padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        # エンコーダ
        d1_1 = self.down1_1(x)  # 28x28 -> 28x28, チャネル数16
        d1_2 = self.down1_2(d1_1)  # 28x28, チャネル数16（スキップ接続用）
        d1_3 = self.down1_3(d1_2)  # 28x28 -> 14x14, チャネル数32

        d2_1 = self.down2_1(d1_3)  # 14x14, チャネル数32
        d2_2 = self.down2_2(d2_1)  # 14x14, チャネル数32
        d2_3 = self.down2_3(d2_2)  # 14x14 -> 7x7, チャネル数64

        b = self.bottom(d2_3)  # ボトム層：7x7, チャネル数64

        # デコーダ
        up1_1 = self.up1_1(b)  # 7x7 -> 14x14, チャネル数32
        cat1 = torch.cat([d1_3, up1_1], dim=1)  # スキップ接続：14x14, チャネル数64
        up1_2 = self.up1_2(cat1)  # 14x14, チャネル数32
        up1_3 = self.up1_3(up1_2)  # 14x14, チャネル数32

        up2_1 = self.up2_1(up1_3)  # 14x14 -> 28x28, チャネル数16
        cat2 = torch.cat([d1_2, up2_1], dim=1)  # スキップ接続：28x28, チャネル数32
        up2_2 = self.up2_2(cat2)  # 28x28, チャネル数16
        up2_3 = self.up2_3(up2_2)  # 28x28, チャネル数16

        out = self.final(up2_3)  # 28x28, チャネル数1（最終出力）
        return out