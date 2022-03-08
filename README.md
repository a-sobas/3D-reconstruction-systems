# **Systems for reconstructing 3D objects from a single image**

Content:
1. General information
2. 3D object reconstruction system in voxel representation <br>
  2.1 Results <br>
  2.2 How it works
3. 3D object reconstruction system in mesh representation <br>
  3.1 Results <br>
  3.2 How it works
  
## **1. General information**
The project consists of two systems that allow the reconstruction of 3D objects from a single photo, based on the encoder-decoder architecture. The first system allows for the reconstruction of objects in a voxel representation, while the second one in a surface representation.

In both cases, the encoder is a neural network with an architecture from the EfficientNet family. During the learning process, the transfer learning mechanism was used for them.

The decoding modules, on the other hand, are completely different from each other due to different representations of the output objects that require different reconstruction mechanisms.

<img src=https://user-images.githubusercontent.com/101074920/157271610-9f178339-5092-48f0-ac8e-16e6b6454886.png width="500"><br>

The first system was trained to reconstruct objects from the armchair category.

The second system was trained to reconstruct objects from the airplanes category.
  
## **2. 3D object reconstruction system in voxel representation**
  
### **2.1 Results**

<img src=https://user-images.githubusercontent.com/101074920/157133592-ad59ec54-3c23-49e5-bf37-e5797efa3064.png width="500"><br>
Fig. 2.1. Illustration showing: (a) input image for the system, (b) the ground truth object in the voxel representation, (c) the reconstructed object shown from different points of view.

<img src=https://user-images.githubusercontent.com/101074920/157133608-17edc289-1624-40e3-b1b6-65ef9f2ab046.png width="500"><br>
Fig. 2.2. Presentation of the results of the reconstruction of various objects.

<img src=https://user-images.githubusercontent.com/101074920/157133624-9a620aae-3442-40cc-a0cc-a56c5139b9bd.png width="500"><br>
Fig. 2.3. Presentation of the results of the reconstruction for the same object obtained by introducing to the system input its images taken from different points of view.

### **2.2. How it works**

<img src=https://user-images.githubusercontent.com/101074920/157141793-d0e579f9-c80f-426b-ad40-dc5379100366.png width="600"><br>
Fig. 2.4. Illustration of the network architecture for reconstructing objects in a voxel representation.

<img src=https://user-images.githubusercontent.com/101074920/157141535-ae1263c9-1884-4edb-acb6-6e41b87453e1.png width="300"><br>
Fig. 2.5. Presentation of the decoder architecture taking into account the values of the parameters like kernel size, stride, padding, number of channels and the size of the feature maps for the inputs and outputs of individual layers.

## **3. 3D object reconstruction system in mesh representation**

### **3.1 Results**

<img src=https://user-images.githubusercontent.com/101074920/157231053-a9585f90-e5f9-4980-bccf-f0e17c69827f.png width="500"><br>
Fig. 3.1. The illustration shows the process of object reconstruction with the use of a reconstruction system in the mesh representation: (a) input image for the system, (b) ground truth object, (c), (d) the initialization object and its subsequent deformation steps.

<img src=https://user-images.githubusercontent.com/101074920/157141560-c57aa8ff-76e8-40e6-ac48-e8892ac72c6e.png width="500"><br>
Fig. 3.2. The ilustration shows: (a) input image for the system, (b) ground truth object, (c) a reconstructed object shown from different points of view.

<img src=https://user-images.githubusercontent.com/101074920/157231552-56b26d6e-23df-4a92-90a3-b5d6de9ce9c8.png width="500"><br>
Fig. 3.3. An illustration showing the input image and two stages of object reconstruction shown against the background of the ground truth object.

<img src=https://user-images.githubusercontent.com/101074920/157143676-744753f2-5515-4de0-b898-af2515d59d69.png width="500"><br>
Fig. 3.4. Presentation of the effects of the reconstruction of various objects.

<img src=https://user-images.githubusercontent.com/101074920/157141576-6fb3f904-2b19-4a28-9900-963dee2a0bee.png width="500"><br>
Fig. 3.5. Presentation of the effects of the reconstruction for the same object obtained by introducing to the system input its photos taken from different points of view.

### **3.2 How it works**

<img src=https://user-images.githubusercontent.com/101074920/157141626-a0eadc00-7ab6-47b0-b71a-976952987560.png width="600"><br>
Fig. 3.6. A schematic drawing showing the architecture of the 3D object reconstruction system in the representation of meshes.

<img src=https://user-images.githubusercontent.com/101074920/157141631-8a80166a-c636-4312-a271-daa62b3d4ab2.png width="300"><br>
Fig. 3.7. Illustration showing the structure of the mesh deformation module taking into account the size of the data structure between the individual steps.


- 3D object reconstruction system in mesh representation was created based on: <br>
G. Gkioxari, J. Johnson and J. Malik, "Mesh R-CNN," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 9784-9794, doi: 10.1109/ICCV.2019.00988.

- The dataset used for the project was Shapenet <br>
Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li et. al., "ShapeNet: An Information-Rich 3D Model Repository", arXiv:1512.03012, 2015

- Voxelized models and object renderings were obtrained from http://3d-r2n2.stanford.edu/ <br>
Choy, Christopher B and Xu, Danfei and Gwak, JunYoung and Chen, Kevin and Savarese, Silvio, "3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction", Proceedings of the European Conference on Computer Vision ({ECCV}), 2016
