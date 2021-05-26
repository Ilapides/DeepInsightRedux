DeepInsight Redux

Isaac Lapides

✅ Project Report

✅ github link: https://github.com/Ilapides/DeepInsightRedux

✅ video: https://youtu.be/-X-kY4yB764

✅ slides: https://docs.google.com/presentation/d/1ZVrRQKWPfRBMybGA_2Lje5Bll2TZdxdPt7EwH4ib18A/edit?usp=sharing
Introduction

Deep learning is a driving force behind computer vision, natural language processing, bioinformatics, and recommendation systems. These are all powerful tools which shape and reshape the modern world. Unsurprisingly, research into deep learning is well-funded and moves at a fast pace. Deep learning is so named because it involves the layering of artificial neural networks to form a larger structure. Building blocks of layerings come together in classes known as architectures. One of the most well-known of these architectures are convolutional neural networks. They get their name from the convolution operation, which combines several elements within a section of matrix to produce a single value. Convolutional neural networks (henceforth CNNs) are most famous because of their breakthroughs in computer vision, where they largely outperform many other deep learning architectures. In part due to their well-deserved hype, they continue to be researched and developed, leading to new innovations in industry and academia. 

However, the convolutional basis of CNNs only becomes a strength on input data formatted in matrices, most commonly images. One problem statement then becomes as follows: how can the utility of CNNs be harnessed to classify originally non-image data? Stated another way, how can input vectors be converted to input matrices? This question was recently addressed by a paper called DeepInsight (Sharma et al., 2019). They used a preprocessing algorithm to transform non-image data reliably into pictures, which a CNN then reliably classified with high performance. The work in this paper was done in MATLAB, however, and most of the implementation details were not included. This then presents a second problem: can the DeepInsight algorithm be reimplemented in Python? Furthermore, what kind of success is possible with this process on different, smaller datasets?

We present a reimplementation of the DeepInsight pipeline in Python. We use this reimplementation to preprocess several cancer microarray datasets, and then train CNNs to classify them. We then compare them to the baselines for these datasets. Our hypothesis is that this process will be competitive with these baselines. 

Related Work

The foundation of deep learning began with artificial neural networks. Inspired by natural neural networks (i.e., those comprising brains and nervous systems), the first abstract models for artificial neural networks were formulated in the mid-20th century.  Since then, work on organic neural networks (particularly mammalian visual systems) has further paved the way for how artificial neural networks might work. In particular, neuroscience research suggests that neurons and small neural networks act as very simple feature detectors (e.g. of a horizontal line), and that these feature detectors cascade into more and more complex feature detectors until eventually the visual system beholds an entire image. In their foundational paper, LeCun et al. (1998) introduced convolutional neural networks with a similar process. They showed that by using convolutional layers (i.e., not fully-connected layers) small features could be detected and built up to recognizing larger images.

Another key piece of the puzzle to our problem is the issue of dimensionality reduction. In machine learning, we must regularly contend with the curse of dimensionality, i.e. the inconvenient, counterintuitive, or downright intractable qualities of high-dimensional spaces. The main tools in contending with high dimensionality are the tools of dimensionality reduction: means by which data with high dimensionality is transformed to occupy a lower-dimensional space. The key to these conversions is the preservation of some intrinsic aspect of the data.  \
The canonical technique for dimensionality reduction is principal component analysis (PCA). This technique performs a linear mapping of the data to a lower (usually 2-) dimensional space, where the lower dimensions are the principal components. These principal components are mappings which maximize the variance of the data. However, sometimes the "intrinsic shape" of the data is not well-represented by a linear model such as PCA. To address this, PCA can be kernelized (e.g. with a Gaussian kernel). By using the kernel trick, _k_PCA can first project the data into a higher-dimensional space in which it is linearly separable before returning the clustered data. We will focus on another well-known method of non-linear dimensionality reduction, _t_-distributed stochastic neighbor embedding (t-SNE). Maaten and Hinton (2008) introduced this technique as a breakthrough in visualizing high-dimensional data in such a way that preserves both local and global structure in lower- (again, usually 2-) dimensional space. As we are concerned with maximally-discernable images, we focus on the use of t-SNE (rather than kPCA) in our approach. 

The DeepInsight approach combines the strength of nonlinear dimensionality reduction and CNNs. The pipeline described in the paper follows four steps. After any necessary initial preprocessing and a test-train split, a non-linear dimensionality reduction technique is applied to the training section of the dataset. Then, a convex hull algorithm is performed by obtaining the minimum bounding rectangle of this set of points in a 2D plot. This rectangle is then rotated, scaled, and translated from Cartesian coordinates into "pixel space," i.e. such that the 2D plot conforms to a square. This produces a map of each feature location, and serves as a template for the conversion of input vectors into matrices. The relative component values of each feature vector are then mapped as pixel intensities to their respective locations. This then produces an image for each input vector.

The Curated Microarray Database (CuMiDa) (Feltes, 2019) is a handpicked repository of cancer microarray datasets for machine learning. This effort was motivated by the scattered and disorganized nature of cancer microarray datasets, which the authors sought to overcome by curating, organizing, and homogenizing specific datasets appropriate for machine learning tasks. Notably, these datasets already have several 3-fold cross validation benchmarks against which we can compare our results. 

Approach

Our ultimate approach was the necessary result of several pivots, failures, and compromises. We will discuss only the final iteration of our approach here; for earlier iterations, see later sections. We wrote code in Python 3 in a Jupyter notebook using a custom Conda environment on Windows. We used the PyTorch library to write our CNN. Additionally, we used the parallel computing platform CUDA for use in training our CNNs on an NVIDIA GPU. This last step was grandfathered in from an earlier iteration of our approach; in the current iteration it does not significantly speed model training. 

First, we sourced several datasets from CuMiDa. GSE14520 was for liver cancer, with 347 samples, 22278 genes, and 2 classes. GSE14520 had benchmarks of .97 for SVM, .8 for MLP, .92 for DT, .96 for NB, .96 for RF, and .93 for KNN. GSE28497 was for leukemia, with 281 samples, 22284 genes, and 7 classes. Benchmarks were .88 for SVM, .72 for MLP, .73 for DT, .78 for NB, .79 for RF, and .7 for KNN. Finally, GSE45827 was for breast cancer, with 151 samples, 54676 genes, and 6 classes. Benchmarks were .94 for SVM, .58 for MLP, .8 for DT, .93 for NB, .95 for RF, and .8 for KNN. We performed a train-test split with test_size=.2 on each dataset.

Next, we implemented the DeepInsight pipeline according to the specifications laid out by the authors. There was some difficulty in translating concepts from MATLAB. One part of this was implementing the authors' pre-pipeline normalizations. Subsequently, we implemented the DeepInsight image production algorithm. We used this with sklearn.manifold.TSNE, and in following with the paper's methods, used cosine similarity as our metric. From here, we applied the image production algorithm fit to the t-SNE results and saved the images in a numpy archive.

Finally, we wrote a simple CNN with two convolutional layers, two max-pooling layers, and three fully connected layers. For our criterion object, we used nn.CrossEntropyLoss, and used stochastic gradient descent. The batch size listed in the supplementary materials for DeepInsight is 128, but we chose to have only one batch (i.e., set batch size greater than number of samples), given how small our computational demands were.   

Evaluation

We were able to successfully use the DeepInsight pipeline to convert our original datasets into images. These images passed a consistency sanity check. Furthermore, feature heat maps resembled those in the DeepInsight paper. However, our quantitative experimental results were very poor. For each dataset, our accuracy was never significantly better than chance. For this reason, we will not present extensive statistics. 

Discussion

Clearly, these are grim results. They are made even more so by the fact that in this autopsy we can only vaguely gesture at several factors. The one element of clarity we have is that we know from the original paper that the DeepInsight method works (on appropriate data) and that the datasets we used have high baselines with other classifiers. This leaves either our DeepInsight implementation or our CNN. The most optimistic explanation is that our (ultimate) datasets had just too few samples to reliably learn a CNN. This is certainly true, but unfortunately we must place much of the blame on our inability to construct a CNN with appropriate hyperparameters. Hyperparameter tuning was symmetrically inhibited by the relative impossibility of model success given the paucity of the dataset. If we were training a CNN without images produced by the DeepInsight pipeline, we might have been able to use image augmentation, a technique by which more training images are artificially created from existing instances. However, this was not possible for us, as DeepInsight heavily constrained the shape of each image, and image augmentation would not have produced valid matrices, and thus would fail in upsampling. We attempted to source larger cellular / genetic datasets, but the only sufficiently large ones we could find were incompatible with some part of our setup, if not Python entirely. 

Lessons Learned

Attempting to force our initial dataset (predicting next-day rain based on data from Australian weather stations) into the DeepInsight pipeline taught us a lot about strategies for transforming categorical variables, non-linear dimensionality reduction, and how to build in sanity checks at different stages of the process. We learned a lot about the inner workings of CNNs, particularly why the convolution operation can be advantageous over the learning performed by dense layers. We also learned about what heuristics to use when doing manual hyperparameter tuning. 

PyTorch gave us many errors, and it forced us to work through documentation page by page. Much of its appeal as a deep learning framework is that it is imperative, well-wrapped, and accessible to beginners. While this was certainly welcome to us, it initially hid much of the complexity that we were forced to reckon with when debugging. Moreover, moving from dataset to dataset would present entirely new errors to debug and thus issues to understand. Although it was ultimately not useful, CUDA was simple to use after initial setup.	

One of the most important things we learned was about the "project life cycle," namely that problem formulation has to be done precisely, and project proposals must have concrete and testable inputs each step of the way. This is especially true for machine learning, given that it is a field where the actual, quantifiable constraints and dimensions of the data make a profound difference on how a problem scales or integrates with different techniques. The fatal flaw of this project was that our original problem formulation was neither focused on practical results nor were the components — the dataset, the pipeline, the chosen CNN architecture — appropriate for each other. Most of the work done on this project was done on making one compromise or another, substituting each element one by one until the initial goal was forgotten. 

Challenges faced

We faced many challenges in this project. The first was how to best represent the Australian weather data as a numerical array. We explored several different options, but none of these options would cluster into distinct groups with dimensionality reduction algorithms, most significantly t-SNE. The global structure was often preserved in one form or another, but there was no local cluster structure to speak of. Losses were ultimately cut and this dataset abandoned. Future research should investigate if poorly-separable classes can still perform well with DeepInsight to CNN. 

Another challenge was environment setup. We had done previous work in Windows Subsystem for Linux, but wished to use CUDA. We attempted to install the preview version of CUDA for WSL, which required installing the latest OS release on the Windows Insider Dev Channel. We foolishly did this without backing up or having another OS installed. This experimental build very quickly broke, in which we lost unsaved work on top of the time spent reinstalling Windows. We then opted to install CUDA on Windows, which was ultimately successful but required jumping through several hoops. 

A major challenge was interpreting the DeepInsight pipeline from the paper. This was confusing at several parts, from when to normalize to how to convert from the t-SNE plot to the feature matrix template. Most challenging was interpreting the supplemental material, particularly on CNN hyperparameters. We tried to incorporate varying degrees of those hyperparameters in our models, but after a certain point it did not make sense to include them without really using them.

One challenge of working alone was that none of the work could be parallelized. This ended up being very inefficient at several points when two components had to be rewritten in a row to figure out if their combination would work. If this were a group project, this overhead would have been reduced due to finding pitfalls sooner.

One of the largest challenges came from our assumptions about the suitability of our data. These were challenged too infrequently and too late, so that with each pivot to a new dataset option, more of the pipeline and CNN had to be changed just to run without errors. Future work we do will carefully consider dataset sizes first to make sure what we are using is appropriate for the problem.

We failed to achieve several goals from our original project proposal. First and foremost, we had to discard our original dataset. Although our focus was the pipeline rather than the data, this movement away from our original dataset represents a serious failure. Furthermore, as our ultimate substitution was a series of datasets with pre-certified baselines, we also did not pursue independent measurement of these baselines. Most crucially, we failed to train our CNN classifier with any measure of success. Though we did implement the DeepInsight technique of producing images/matrices from vectors, this complete lack of success with our CNN means that we cannot truly evaluate the performance of our DeepInsight transformations. 

References

Feltes, B. C., Chandelier, E. B., Grisci, B. I., & Dorn, M. (2019). CuMiDa: An Extensively  \
	Curated Microarray Database for Benchmarking and Testing of Machine Learning  \
	Approaches in Cancer Research. _Journal of Computational Biology_, _26_(4), 376–386.  \
	https://doi.org/10.1089/cmb.2018.0238 

Lecun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to  \
	document recognition. _Proceedings of the IEEE_, _86_(11), 2278–2324.  \
	https://doi.org/10.1109/5.726791 

Sharma, A., Vans, E., Shigemizu, D., Boroevich, K. A., & Tsunoda, T. (2019). DeepInsight: A  \
	methodology to transform a non-image data to an image for convolution neural network  \
	architecture. _Scientific Reports_, _9_(1). https://doi.org/10.1038/s41598-019-47765-6 

Van Der Maaten & Hinton (2008). Visualizing Data using t-SNE. _Journal of Machine Learning  \
	Research_, 86(9) https://jmlr.org/papers/v9/vandermaaten08a.html
