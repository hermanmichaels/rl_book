# About

This repository is an addition to my [post series](https://medium.com/@hrmnmichaels) covering [Sutton's Reinforcement Learning Book](http://incompleteideas.net/book/RLbook2020.pdf). 

It contains code reproducing all the shown results, and will grow along with the post series.

# Usage

## Installation

[Install poetry](https://medium.com/towards-data-science/dependency-management-with-poetry-f1d598591161), then run poetry install.

## Bandits

    python bandits.py

## Grid World

    python grid_world.py --method {METHOD}

# Change History
- 01/07/2025: Refactor RL framework and introduce multi-player envs and games. DP methods are left untouched since they are problemtic to fit into the new framework and hard to extend to multi-player games. MC with Exploring Starts and the custom non-policy MC method are removed, since they were for demonstration purposes only anyways. Prioritized Sweeping is removed since it is not compatible with the new framework.
