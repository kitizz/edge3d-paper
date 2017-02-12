Boundaries are Fleeting: Texture is Forever
===========================================

Reference
---------
If you use or reference this work, please cite the following paper:
 - **Boundaries are Fleeting: Texture is Forever - Moving Past Brightness Constancy**; C. Ham, S. Singh, S. Lucey; In WACV 2017

Citations
---------
For the paper we used Piotr Dollár's Structured Forests Edge Detector. Source can be found at https://github.com/pdollar/edges. Reference:
 - **Structured Forests for Fast Edge Detection**; P. Dollár and C. Zitnick; In ICCV 2013


Foreword
--------
In the spirit of open source and enabling reproducibility of academic work, this code is almost the exact code used in the paper. It has had some minor refactoring and cleaning up of comments. And so, it is quite raw, and rough around the edges.

I will release a version 2 in the coming year as I rewrite parts of it to be integrated with my final thesis.


Requirements
------------
This project has been developed for **Python 3**. You may be able to tweak it to work with Python 2.7, but there are no guarantees that it will work.

Install the basic Python modules required with:
```
pip install -r requirements.txt
```

This project also relies on Numba to accelerate the logic and enable multithreaded processing in Python. Refer to the [Numba website](http://numba.pydata.org/) for installation instructions.

For your own dataset, if you wish to use the Structured Forest Edge Detector, source can be found at https://github.com/pdollar/edges. Alternatively, you could use any edge detector you like and place the images in a subdirectory of the sequence called `edges` (see Usage instructions below). To work, the functions `edgesTrain` and `edgesDetect` must be added to the path in Matlab.


Usage
-----
To process a sequence in a directory, use Pipeline.py
```
python Pipeline.py <path/to/sequence>
```

Sequence datasets used in the paper can be downloaded in the [Release section](https://github.com/kitizz/edge3d-paper/releases) of the GitHub project.

At minimum a sequence directory should contain the following files:
```
sequence_name
  |- cam.yaml
  |- config.yaml
  |- rects.yaml
  |- {edges/, seq/, video.mp4}
```

`edges` should be a directory of images with the edge response map of the original sequence. If Pipeline cannot find this directory, it will attempt to generate the edge maps using the Structured Forest Edge Detector in Matlab using images in the `seq` directory. If `seq` doesn't exist, it will generate the images from `video.mp4` using `avconv`.

# Camera calibration
`cam.yaml` should contain the intrinsic camera calibration of the sequence. An example:
```
fx: 1100.0
fy: 1100.0
cx: 640.0
cy: 360.0
```

# Configuration
See the paper datasets for example configuration files. They set the values of the `Config` object passed to different stages of the pipeline. Important configurations are:

- **frame_support**: The temporal window radius when looking for neighbor edge segments.
- **min_inlier**: The minimum fraction of inliers in the RANSAC fitting for an edge to be considered good quality.
- **min_neighbors**: The minimum number of neighbors to reciprocate a correspondence for an edge to be considered good quality

# Bounding Rectangles
I will soon update the code to make this optional. This currently a nice way to constrain the bounds of the voxel grid used to speed up look-up times of edge rays. It has no affect on the results, it simply helps increase the computational efficiency. An example:
```
rects:
	- frame: 500
	  bounds: [300, 150, 350, 200]
	- frame: 950
	  bounds: [200, 400, 250, 450]
```

Where bounds is a simple bounding box defined by the upper left and lower right vertices: `[x1, y1, x2, y2]`.
