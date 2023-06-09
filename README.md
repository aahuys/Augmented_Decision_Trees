## Code
This repository contains all code surrounding the construction and evaluation of the augmented decision trees. To construct and build a tree, the main file should be ran. All code is commented and different parameters or trees can easily be build.

InitData.py contains all the code for the initialization of the MovieLens dataset.

DecisionTrees.py contains all the code surrounding the construction of the decision trees.

Forest.py contains all the code surrounding the construction of the forests.

ParallelTrees.py contains all the code surrounding the construction of the trees in a parallel way.

Evaluation.py contains all the code for evaluating the trees or forests.

OnlineEvaluation.py contains all the code for the online evaluation on data received by the online platform and stored in the Results_online_evaluation. The binary_tree.pkl file was used to construct the tree of the online platform.

Dataset_Analyzation.ipynb contains a notebook with all graphs used to describe the dataset.

Offline evaluation.ipynb contains all the code for the graphs used to analyze the trees. A template is to tune parameters for a tree is presented as well.

Online_evaluation.ipynb contains all the graphs used to analyze the binary tree.

A visualization of a large decision tree is stored in decision_tree.pdf
