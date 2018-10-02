# Target-driven Visual Navigation Model using Deep Reinforcement Learning
This is implementation of http://web.stanford.edu/~yukez/papers/icra2017.pdf in PyTorch. It attempts to achieve the same results as the Tensorflow implementation, which can be found here: https://github.com/zfw1226/icra2017-visual-navigation.

![THOR scene samples](http://web.stanford.edu/~yukez/images/img/thor_examples.png "THOR scene samples")

## Introduction

This repocitory provides a Tensorflow implementation of the deep siamese actor-critic model for indoor scene navigation introduced in the following paper:

**[Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning](http://web.stanford.edu/~yukez/papers/icra2017.pdf)**
<br>
[Yuke Zhu](http://web.stanford.edu/~yukez/), Roozbeh Mottaghi, Eric Kolve, Joseph J. Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi
<br>
[ICRA 2017, Singapore](http://www.icra2017.org/)

## Setup and run
This code is implemented in [Pytorch 0.4](https://pytorch.org/). It uses [Docker](http://docker.com/) to automate instalation process. In order to run this code, I recommend pulling it from my [dockerhub repository](http://hub.docker.com/kulhanek/visual-navigation-agent-pytorch).

In order to start training, run those commands:
```
git clone https://github.com/jkulhanek/visual-navigation-agent-pytorch
docker-compose run train
```

## Scenes
To facilitate training, we provide [hdf5](http://www.h5py.org/) dumps of the simulated scenes. Each dump contains the agent's first-person observations sampled from a discrete grid in four cardinal directions. To be more specific, each dump stores the following information row by row:

* **observation**: 300x400x3 RGB image (agent's first-person view)
* **resnet_feature**: 2048-d [ResNet-50](https://arxiv.org/abs/1512.03385) feature extracted from the observations
* **location**: (x, y) coordinates of the sampled scene locations on a discrete grid with 0.5-meter offset
* **rotation**: agent's rotation in one of the four cardinal directions, 0, 90, 180, and 270 degrees
* **graph**: a state-action transition graph, where ```graph[i][j]``` is the location id of the destination by taking action ```j``` in location ```i```, and ```-1``` indicates collision while the agent stays in the same place.
* **shortest_path_distance**: a square matrix of shortest path distance (in number of steps) between pairwise locations, where ```-1``` means two states are unreachable from each other.
## Acknowledgements
I would like to acknowledge the following references that have offered great help for me to implement the model.
* ["Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016](https://arxiv.org/abs/1602.01783)
* [David Silver's Deep RL course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [muupan's async-rl repo](https://github.com/muupan/async-rl/wiki)
* [miyosuda's async_deep_reinforce repo](https://github.com/miyosuda/async_deep_reinforce)
* [miyosuda's async_deep_reinforce repo](https://github.com/miyosuda/async_deep_reinforce)
* [Pytorch A3C implementation repo](https://github.com/ikostrikov/pytorch-a3c)


## Citation
Please cite our ICRA'17 paper if you find this code useful for your research.
```
@InProceedings{zhu2017icra,
  title = {{Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning}},
  author = {Yuke Zhu and Roozbeh Mottaghi and Eric Kolve and Joseph J. Lim and Abhinav Gupta and Li Fei-Fei and Ali Farhadi},
  booktitle = {{IEEE International Conference on Robotics and Automation}},
  year = 2017,
}
```

## License
MIT
