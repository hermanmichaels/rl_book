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


class ConnectFourCNN(nn.Module):
    """
    Lightweight CNN for Connect4 Q-values.
    Input: [bs, 2, 6, 7] (player-to-move pieces, opponent pieces)
    Output: [bs, 7] Q-values for columns.
    """

    def __init__(self, num_actions: int = 7, hidden: int = 64, pooled: int = 2) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)

        # Keep coarse spatial layout (2x2 works well for 6x7)
        self.pool = nn.AdaptiveAvgPool2d((pooled, pooled))
        self.fc = nn.Linear(hidden * pooled * pooled, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)
