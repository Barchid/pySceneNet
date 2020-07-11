import scenenet_pb2 as sn
import os
import numpy as np
from PIL import Image

import argparse

NYU_13_CLASSES = [(0, 'Unknown'),
                  (1, 'Bed'),
                  (2, 'Books'),
                  (3, 'Ceiling'),
                  (4, 'Chair'),
                  (5, 'Floor'),
                  (6, 'Furniture'),
                  (7, 'Objects'),
                  (8, 'Picture'),
                  (9, 'Sofa'),
                  (10, 'Table'),
                  (11, 'TV'),
                  (12, 'Wall'),
                  (13, 'Window')
                  ]

NYU_40_CLASSES = [
    (0, 'Unkown'),
    (1, "Wall"),
    (2, "Floor"),
    (3, "Cabinet"),
    (4, "Bed"),
    (5, "Chair"),
    (6, "Sofa"),
    (7, "Table"),
    (8, "Door"),
    (9, "Window"),
    (10, "BookShelf"),
    (11, "Picture"),
    (12, "Counter"),
    (13, "Blinds"),
    (14, "Desks"),
    (15, "Shelves"),
    (16, "Curtain"),
    (17, "Dresser"),
    (18, "Pillow"),
    (19, "Mirror"),
    (20, "Floor-mat"),
    (21, "Clothes"),
    (22, "Ceiling"),
    (23, "Books"),
    (24, "Refrigerator"),
    (25, "Television"),
    (26, "Paper"),
    (27, "Towel"),
    (28, "Shower-curtain"),
    (29, "Box"),
    (30, "Whiteboard"),
    (31, "Person"),
    (32, "NightStand"),
    (33, "Toilet"),
    (34, "Sink"),
    (35, "Lamp"),
    (36, "Bathtub"),
    (37, "Bag"),
    (38, "Other-structure"),
    (39, "Other-furniture"),
    (40, "Other-prop")
]

colour_code = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0.9137, 0.3490, 0.1882],  # BOOKS
                        [0, 0.8549, 0],  # CEILING
                        [0.5843, 0, 0.9412],  # CHAIR
                        [0.8706, 0.9451, 0.0941],  # FLOOR
                        [1.0000, 0.8078, 0.8078],  # FURNITURE
                        [0, 0.8784, 0.8980],  # OBJECTS
                        [0.4157, 0.5333, 0.8000],  # PAINTING
                        [0.4588, 0.1137, 0.1608],  # SOFA
                        [0.9412, 0.1373, 0.9216],  # TABLE
                        [0, 0.6549, 0.6118],  # TV
                        [0.9765, 0.5451, 0],  # WALL
                        [0.8824, 0.8980, 0.7608]])

# NYU_WNID_TO_CLASS = {
#     '04593077': 4, '03262932': 4, '02933112': 6, '03207941': 7, '03063968': 10, '04398044': 7, '04515003': 7,
#     '00017222': 7, '02964075': 10, '03246933': 10, '03904060': 10, '03018349': 6, '03786621': 4, '04225987': 7,
#     '04284002': 7, '03211117': 11, '02920259': 1, '03782190': 11, '03761084': 7, '03710193': 7, '03367059': 7,
#     '02747177': 7, '03063599': 7, '04599124': 7, '20000036': 10, '03085219': 7, '04255586': 7, '03165096': 1,
#     '03938244': 1, '14845743': 7, '03609235': 7, '03238586': 10, '03797390': 7, '04152829': 11, '04553920': 7,
#     '04608329': 10, '20000016': 4, '02883344': 7, '04590933': 4, '04466871': 7, '03168217': 4, '03490884': 7,
#     '04569063': 7, '03071021': 7, '03221720': 12, '03309808': 7, '04380533': 7, '02839910': 7, '03179701': 10,
#     '02823510': 7, '03376595': 4, '03891251': 4, '03438257': 7, '02686379': 7, '03488438': 7, '04118021': 5,
#     '03513137': 7, '04315948': 7, '03092883': 10, '15101854': 6, '03982430': 10, '02920083': 1, '02990373': 3,
#     '03346455': 12, '03452594': 7, '03612814': 7, '06415419': 7, '03025755': 7, '02777927': 12, '04546855': 12,
#     '20000040': 10, '20000041': 10, '04533802': 7, '04459362': 7, '04177755': 9, '03206908': 7, '20000021': 4,
#     '03624134': 7, '04186051': 7, '04152593': 11, '03643737': 7, '02676566': 7, '02789487': 6, '03237340': 6,
#     '04502670': 7, '04208936': 7, '20000024': 4, '04401088': 7, '04372370': 12, '20000025': 4, '03956922': 7,
#     '04379243': 10, '04447028': 7, '03147509': 7, '03640988': 7, '03916031': 7, '03906997': 7, '04190052': 6,
#     '02828884': 4, '03962852': 1, '03665366': 7, '02881193': 7, '03920867': 4, '03773035': 12, '03046257': 12,
#     '04516116': 7, '00266645': 7, '03665924': 7, '03261776': 7, '03991062': 7, '03908831': 7, '03759954': 7,
#     '04164868': 7, '04004475': 7, '03642806': 7, '04589593': 13, '04522168': 7, '04446276': 7, '08647616': 4,
#     '02808440': 7, '08266235': 10, '03467517': 7, '04256520': 9, '04337974': 7, '03990474': 7, '03116530': 6,
#     '03649674': 4, '04349401': 7, '01091234': 7, '15075141': 7, '20000028': 9, '02960903': 7, '04254009': 7,
#     '20000018': 4, '20000020': 4, '03676759': 11, '20000022': 4, '20000023': 4, '02946921': 7, '03957315': 7,
#     '20000026': 4, '20000027': 4, '04381587': 10, '04101232': 7, '03691459': 7, '03273913': 7, '02843684': 7,
#     '04183516': 7, '04587648': 13, '02815950': 3, '03653583': 6, '03525454': 7, '03405725': 6, '03636248': 7,
#     '03211616': 11, '04177820': 4, '04099969': 4, '03928116': 7, '04586225': 7, '02738535': 4, '20000039': 10,
#     '20000038': 10, '04476259': 7, '04009801': 11, '03909406': 12, '03002711': 7, '03085602': 11, '03233905': 6,
#     '20000037': 10, '02801938': 7, '03899768': 7, '04343346': 7, '03603722': 7, '03593526': 7, '02954340': 7,
#     '02694662': 7, '04209613': 7, '02951358': 7, '03115762': 9, '04038727': 6, '03005285': 7, '04559451': 7,
#     '03775636': 7, '03620967': 10, '02773838': 7, '20000008': 6, '04526964': 7, '06508816': 7, '20000009': 6,
#     '03379051': 7, '04062428': 7, '04074963': 7, '04047401': 7, '03881893': 13, '03959485': 7, '03391301': 7,
#     '03151077': 12, '04590263': 13, '20000006': 1, '03148324': 6, '20000004': 1, '04453156': 7, '02840245': 2,
#     '04591713': 7, '03050864': 7, '03727837': 5, '06277280': 11, '03365592': 5, '03876519': 8, '03179910': 7,
#     '06709442': 7, '03482252': 7, '04223580': 7, '02880940': 7, '04554684': 7, '20000030': 9, '03085013': 7,
#     '03169390': 7, '04192858': 7, '20000029': 9, '04331277': 4, '03452741': 7, '03485997': 7, '20000007': 1,
#     '02942699': 7, '03231368': 10, '03337140': 7, '03001627': 4, '20000011': 6, '20000010': 6, '20000013': 6,
#     '04603729': 10, '20000015': 4, '04548280': 12, '06410904': 2, '04398951': 10, '03693474': 9, '04330267': 7,
#     '03015149': 9, '04460038': 7, '03128519': 7, '04306847': 7, '03677231': 7, '02871439': 6, '04550184': 6,
#     '14974264': 7, '04344873': 9, '03636649': 7, '20000012': 6, '02876657': 7, '03325088': 7, '04253437': 7,
#     '02992529': 7, '03222722': 12, '04373704': 4, '02851099': 13, '04061681': 10, '04529681': 7,
# }

NYU_WNID_TO_CLASS = {'01091234': 39, '14845743': 37, '14974264': 25, '15075141': 39,
                     '15101854': 38, '00017222': 39, '00266645': 37, '02676566': 39, '02686379': 37, '02694662': 39,
                     '02738535': 4, '02747177': 37, '02773838': 36, '02777927': 0, '02789487': 11, '02801938': 28,
                     '02808440': 35, '02815950': 21, '02823510': 39, '02828884': 4, '02839910': 37, '02840245': 22,
                     '02843684': 37, '02851099': 12, '02871439': 9, '02876657': 39, '02880940': 39, '02881193': 39,
                     '02883344': 28, '02920083': 3, '02920259': 3, '02933112': 2, '02942699': 39, '02946921': 39,
                     '02951358': 37, '02954340': 20, '02960903': 39, '02964075': 6, '02990373': 21, '02992529': 39,
                     '03001627': 4, '03002711': 4, '03005285': 39, '03015149': 5, '03018349': 2, '03025755': 39,
                     '03046257': 39, '03050864': 28, '03063599': 39, '03063968': 6, '03071021': 39, '03085013': 39,
                     '03085219': 39, '03085602': 24, '03092883': 6, '03115762': 5, '03116530': 11, '03128519': 39,
                     '03147509': 39, '03148324': 38, '03151077': 15, '03165096': 3, '03168217': 4, '03169390': 39,
                     '03179701': 13, '03179910': 39, '03206908': 39, '03207941': 37, '03211117': 24, '03211616': 24,
                     '03221720': 7, '03222722': 7, '03231368': 6, '03233905': 16, '03237340': 16, '03238586': 6, '03246933': 6, '03261776': 39,
                     '03262932': 4, '03273913': 23, '03309808': 20, '03325088': 33, '03337140': 25, '03346455': 39, '03365592': 1, '03367059': 34,
                     '03376595': 4, '03379051': 20, '03391301': 37, '03405725': 38, '03438257': 39, '03452594': 37, '03452741': 37, '03467517': 39,
                     '03482252': 37, '03485997': 37, '03488438': 39, '03490884': 39, '03513137': 39, '03525454': 39, '03593526': 39, '03603722': 39,
                     '03609235': 37, '03612814': 39, '03620967': 6, '03624134': 39, '03636248': 34, '03636649': 34, '03640988': 34, '03642806': 39,
                     '03643737': 39, '03649674': 4, '03653583': 38, '03665366': 34, '03665924': 34, '03676759': 24, '03677231': 39, '03691459': 39,
                     '03693474': 5, '03710193': 37, '03727837': 19, '03759954': 39, '03761084': 39, '03773035': 18, '03775636': 33, '03782190': 24,
                     '03786621': 4, '03797390': 39, '03876519': 10, '03881893': 8, '03891251': 4, '03899768': 4, '03904060': 6, '03906997': 39,
                     '03908831': 39, '03909406': 39, '03916031': 39, '03920867': 4, '03928116': 37, '03938244': 17, '03956922': 37, '03957315': 39,
                     '03959485': 39, '03962852': 3, '03982430': 6, '03990474': 39, '03991062': 39, '04004475': 39, '04009801': 24, '04038727': 38,
                     '04047401': 37, '04061681': 13, '04062428': 5, '04074963': 39, '04099969': 4, '04101232': 37, '04118021': 19, '04152593': 24,
                     '04152829': 24, '04164868': 37, '04177755': 5, '04177820': 4, '04183516': 39, '04186051': 39, '04190052': 14, '04192858': 39,
                     '04208936': 37, '04209613': 37, '04223580': 33, '04225987': 39, '04253437': 39, '04254009': 39, '04255586': 39, '04256520': 5,
                     '04284002': 39, '04306847': 37, '04315948': 39, '04330267': 37, '04331277': 4, '04337974': 37, '04343346': 39, '04344873': 5,
                     '04349401': 39, '04372370': 0, '04373704': 4, '04379243': 6, '04380533': 34, '04381587': 6, '04398044': 39, '04398951': 6,
                     '04401088': 39, '04446276': 32, '04447028': 32, '04453156': 39, '04459362': 26, '04460038': 37, '04466871': 37, '04476259': 39,
                     '04502670': 39, '04515003': 38, '04516116': 39, '04522168': 39, '04526964': 39, '04529681': 39, '04533802': 39, '04546855': 0,
                     '04548280': 39, '04550184': 38, '04553920': 33, '04554684': 37, '04559451': 33, '04569063': 39, '04586225': 39, '04587648': 8,
                     '04589593': 8, '04590263': 8, '04590933': 4, '04591713': 39, '04593077': 4, '04599124': 39, '04603729': 6, '04608329': 13,
                     '06277280': 24, '06410904': 22, '06415419': 22, '06508816': 25, '06709442': 39, '08266235': 6, '08647616': 4, '20000004': 3,
                     '20000006': 3, '20000007': 3, '20000008': 2, '20000009': 2, '20000010': 2, '20000011': 2, '20000012': 2, '20000013': 2,
                     '20000015': 4, '20000016': 4, '20000018': 4, '20000020': 4, '20000021': 4, '20000022': 4, '20000023': 4, '20000024': 4,
                     '20000025': 4, '20000026': 4, '20000027': 4, '20000028': 5, '20000029': 5, '20000030': 5, '20000036': 6, '20000037': 6,
                     '20000038': 6, '20000039': 6, '20000040': 6, '20000041': 6}

# These functions produce a file path (on Linux systems) to the image given
# a view and render path from a trajectory.  As long the data_root_path to the
# root of the dataset is given.  I.e. to either val or train


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def photo_path_from_view(render_path, view):
    photo_path = os.path.join(render_path, 'photo')
    image_path = os.path.join(photo_path, '{0}.jpg'.format(view.frame_num))
    return os.path.join(data_root_path, image_path)


def instance_path_from_view(render_path, view):
    photo_path = os.path.join(render_path, 'instance')
    image_path = os.path.join(photo_path, '{0}.png'.format(view.frame_num))
    return os.path.join(data_root_path, image_path)


def depth_path_from_view(render_path, view):
    photo_path = os.path.join(render_path, 'depth')
    image_path = os.path.join(photo_path, '{0}.png'.format(view.frame_num))
    return os.path.join(data_root_path, image_path)


def save_class_from_instance(instance_path,
                             class_path,
                             class_NYUv2_colourcode_path,
                             mapping):
    instance_img = np.asarray(Image.open(instance_path))
    class_img = np.zeros(instance_img.shape)
    h, w = instance_img.shape

    # class_img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # r = class_img_rgb[:, :, 0]
    # g = class_img_rgb[:, :, 1]
    # b = class_img_rgb[:, :, 2]

    for instance, semantic_class in mapping.items():
        class_img[instance_img == instance] = semantic_class
        # r[instance_img == instance] = np.uint8(
        #     colour_code[semantic_class][0]*255)
        # g[instance_img == instance] = np.uint8(
        #     colour_code[semantic_class][1]*255)
        # b[instance_img == instance] = np.uint8(
        #     colour_code[semantic_class][2]*255)

    # class_img_rgb[:, :, 0] = r
    # class_img_rgb[:, :, 1] = g
    # class_img_rgb[:, :, 2] = b

    class_img = Image.fromarray(np.uint8(class_img))
    #class_img_rgb = Image.fromarray(class_img_rgb)
    class_img.save(class_path)
    # class_img_rgb.save(class_NYUv2_colourcode_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root_path", type=str, default='/se3netsproject/val',
                        help="root folder where the data lies")

    parser.add_argument("protobuf_path", type=str, default='/se3netsproject/scenenet_rgbd_val.pb',
                        help="increase output verbosity")

    args = parser.parse_args()

    data_root_path = args.data_root_path
    protobuf_path = args.protobuf_path
    mkdir(os.path.join(data_root_path, 'class40'))

    trajectories = sn.Trajectories()
    try:
        with open(protobuf_path, 'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet protobuf data not found at location:{0}'.format(
            data_root_path))
        print('Please ensure you have copied the pb file to the data directory')

    print('Number of trajectories:{0}'.format(len(trajectories.trajectories)))
    for traj in trajectories.trajectories:

        instance_class_map = {}
        for instance in traj.instances:
            instance_type = sn.Instance.InstanceType.Name(
                instance.instance_type)
            if instance.instance_type != sn.Instance.BACKGROUND:
                instance_class_map[instance.instance_id] = NYU_WNID_TO_CLASS[instance.semantic_wordnet_id]

        '''
        The instances attribute of trajectories contains all of the information
        about the different instances.  The instance.instance_id attribute provides
        correspondences with the rendered instance.png files.  I.e. for a given
        trajectory, if a pixel is of value 1, the information about that instance,
        such as its type, semantic class, and wordnet id, is stored here.
        For more information about the exact information available refer to the
        scenenet.proto file.
        '''

        '''
        The views attribute of trajectories contains all of the information
        about the rendered frames of a scene.  This includes camera poses,
        frame numbers and timestamps.
        '''

        for view in traj.views:
            #print(protobuf_path)
            print(photo_path_from_view(traj.render_path, view))

            instance_path = instance_path_from_view(traj.render_path, view)
            instance_path = os.path.normpath(instance_path)
            instance_path_splits = instance_path.split(os.sep)
            # pb_num = instance_path_splits[3]
            # dir_num = instance_path_splits[4]

            # create class_dir if not exist
            class_dir = os.path.join(
                instance_path_splits[0], instance_path_splits[1], instance_path_splits[2], instance_path_splits[3], 'class40')
            mkdir(class_dir)

            # create file of class
            class_path = os.path.join(
                class_dir, '{0}.png'.format(view.frame_num))

            # class_path = os.path.join(data_root_path, 'class40', 'semantic_class40_{0}_{1}_{2}.png'.format(
            #     pb_num, dir_num, view.frame_num))

            print(class_path)
            class_NYUv2_colourcode_path = class_path.replace(
                'class13', 'class13colour')

            save_class_from_instance(instance_path_from_view(traj.render_path, view),
                                     class_path,
                                     class_NYUv2_colourcode_path,
                                     instance_class_map)
