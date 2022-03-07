import os
from vtkplotter import *
from vedo import Mesh
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pytorch3d.ops import cubify


def show_predicted_voxels(predicted_voxels, id):
    voxels = cubify_(predicted_voxels)

    s_id, m_id, i_id = id.split('-')
    img_path = os.path.join(
        os.getcwd(), 
        'preprocessed_datasets', 
        s_id, 
        m_id, 
        'images', 
        i_id + '.png'
    )

    img = mpimg.imread(img_path)

    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    vp = Plotter(title='predicted voxels', axes=0)
    vp.show(voxels, azimuth=45, elevation=30, roll=0).close()


def show_predicted_meshes(meshes, id):
    s_id, m_id, i_id = id.split('-')
    img_path = os.path.join(
        os.getcwd(), 
        'preprocessed_datasets', 
        s_id, 
        m_id, 
        'images', 
        i_id + '.png'
    )

    img = mpimg.imread(img_path)

    plt.imshow(img)

    plt.tight_layout()
    plt.show()
    vp = Plotter(shape=[2, 1],
                 axes=0
                 )
    if isinstance(meshes, list):
        meshes = meshes
        ms = []
        for i, mesh in enumerate(meshes):
            mesh = mesh.detach()
            verts = mesh.verts_list()[0]
            faces = mesh.faces_list()[0]

            ms.append(Mesh([verts, faces], alpha=1, c='grey'))
            ms[i].lineColor('black')

        vp.show(ms[0], azimuth=0, elevation=0, roll=0, at=0)
        vp.show(ms[1], azimuth=150, elevation=55, roll=0, at=1, interactive=True).close()


def cubify_(voxels):
    cubified_voxels_1 = cubify(voxels.unsqueeze_(0), 0.2)
    verts = cubified_voxels_1.verts_list()[0]
    faces = cubified_voxels_1.faces_list()[0]
    mesh = Mesh([verts, faces], c='grey')

    return mesh

