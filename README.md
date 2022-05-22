# Description
This is a python project using computer vision to solve sudoku puzzles in real time via camera feed. The grid is tracked, machine learning is used to read the numbers on the grid, the puzzle is solved, and the solution is written to the image and warped such that the numbers appear in their correct position on the grid shown in the camera.

`cnet.p` is a state dictionary containing the pre-trained neural network used to detect digits. To train the network manually, the function `train(num)` can be used, where `num` is the number of epochs used in the training process. A higher `num` will take longer, but yield more accurate results. A `num` of 20 is a good balance between time and accuracy.

The `assets` folder contains the training images used to train the network. Additional images can be added for more accurate results.

# Usage
In command line:
`python solve.py`
A video window will open up, and any found sudoku grids will have their solutions overlayed on top of them in the video feed.

# Dependencies
[py-sudoku](https://pypi.org/project/py-sudoku/)

[OpenCV](https://opencv.org/)

[PyTorch](https://pytorch.org/)
