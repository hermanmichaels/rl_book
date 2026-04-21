import torch
import torch.nn as nn
import torch.nn.functional as F


class GridWorldCNN(nn.Module):
    """Simple CNN to process rasterized GridWorld images and output Q values."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_actions)

    def forward(self, x: torch.Tensor):
        """Forward call.

        Args:
            x: input tensor [bs, C, H, W]

        Returns:
            Q values [bs, num_actions]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TicTacToeMLP(nn.Module):
    """Simple CNN to process rasterized GridWorld images and output Q values."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor):
        """Forward call.

        Args:
            x: input tensor [bs, C, H, W]

        Returns:
            Q values [bs, num_actions]
        """
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = x + residual
        return F.relu(x)


class ConnectFourCNN(nn.Module):
    """
    CNN to parse ConnectFour scenes.

    Input:
        [B, 2, 6, 7]
        channel 0 = current player's stones
        channel 1 = opponent's stones

    Output:
        [B, 7] Q-values for each column
    """

    def __init__(
        self,
        num_actions: int = 7,
        channels: int = 64,
        fc_hidden: int = 128,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_conv = nn.Conv2d(2, channels, kernel_size=3, padding=1)

        self.res1 = ResidualBlock(channels)
        self.res2 = ResidualBlock(channels)
        self.res3 = ResidualBlock(channels)

        self.fc1 = nn.Linear(channels * 6 * 7, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_actions)

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_conv(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x