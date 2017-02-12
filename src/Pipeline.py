import sys
import os
from subprocess import run

from tools import Preprocessing
import RaySoup
import RayCloud
import Config
import ClassifyEdges
import OcclusionEdges
import PersistentEdges


def matlab_edges(seq, nthreads=4):
    # matlab -nosplash -nodesktop -r "seq='../data/bottle_02/seq/';out='../data/bottle_02/edges/';parpool(4);FastEdges"
    script = "seq='{seq}/seq/';out='{seq}/edges/';parpool({nthreads});FastEdges;exit".format(seq=seq, nthreads=nthreads)
    cmd = ['matlab', '-nosplash', '-nodesktop', '-r', script]

    edgepath = os.path.join(seq, 'edges')
    if not os.path.exists(edgepath):
        run(cmd, check=True, stdout=sys.stdout)


def process_sequence(seq):
    '''
    seq: Directory of sequence
    '''
    config_path = os.path.join(seq, 'config.yaml')
    config = Config.load(config_path)

    skip = config.frame_subsample
    sub = 'rays_{}'.format(skip)

    # Make images and edges first if they don't exists
    edge_path = os.path.join(seq, 'edges')
    if not os.path.exists(edge_path):
        # Extract images from video first
        im_path = os.path.join(seq, 'seq')
        if not os.path.exists(im_path):
            Preprocessing.video_to_images(seq)

        matlab_edges(seq)

    cloud_path = os.path.join(seq, sub)
    if not os.path.exists(cloud_path):
        # Extract and vectorize the edges as "edge rays"
        RaySoup.extract_rays(seq, sub=sub, skip=skip, imtype='png')
        # Build a voxel grid for fast lookups
        RaySoup.build_voxel_grid(seq, sub)

    cloud = RayCloud.load( cloud_path )
    config.eps = 1/cloud.cam[0]

    # Classify edges and save out
    # ClassifyEdges.detect(seq, sub, cloud, config, imtype='png')

    # Reconstruct Non-Persistent Edges
    OcclusionEdges.reconstruct(seq, sub, cloud, config)

    # Reconstruct Persistent Edges
    PersistentEdges.reconstruct(seq, sub, cloud, config)


if __name__ == '__main__':
    seq = sys.argv[1]
    process_sequence(seq)
