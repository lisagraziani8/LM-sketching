# Language Modeling-like approach to Sketching

We proposed a model which generates sketches, in which sketches are represented as sequences of segments. This means that segments are ordered with respect to time. We train an RNN using such sequences and then we use the RNN to generate new sketches. After training we provide the network some initial segments and let it complete the sketch, generating one segment at each time step. In this approach the sketch creation occurs as in natural language processing where the next word is predicted. In this case, unlike the language, the primitive is a segment.

Given an image of a sketch $h \times w$ with $h=w$, each segment of the image is represented by four integer numbers $(x_1,y_1),(x_2,y_2)$, namely the endpoints coordinates.
The segments have to be collected following given ordering criteria, simulating in what order a person could draw lines of a sketch. We extracted segments from images using a contour-based extractor. First edges are detected on the image, and contours are extracted from them. Then the contours are approximated to obtain polygons.

We approximated the coordinates building over the image a grid $g \times g$, starting from 0 and ending in $g \cdot g - 1$. In this way each point is transformed into an integer number of the grid.
So each segment is represented by two integer numbers $\alpha, \beta \in \{0,..., g \cdot g - 1\}$, where $\alpha$ is the number on the grid on which the point $(x_1,y_1)$ lies (segment start) and $\beta$ where lies the point $(x_2,y_2)$ (segment end).

The segments of the image are represented with a matrix $X \in G^{(N+2) \times 2}$, where $G=\{0,..., g \cdot g + 1\}$, $N$ is the maximum number of segments that a sketch can contain, and the $+2$ rows stay for a symbol of start, that is $g \cdot g$, and a symbol of stop $g \cdot g +1$.


### Code

1) *build_matrix.py* 

From sketch images extract segments. 
Sketch are represented with segments endpoints [x0,y0,x1,y1]. 

2) *grid_build_matrix.py*

Convert segments from [x0,y0,x1,y1] into [alpha, beta].

3) *grid_trainA.py*

Train sketch model to predict the next segments.

