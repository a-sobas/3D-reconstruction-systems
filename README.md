# 3D-reconstruction-systems

W ramach projektu powstały dwa systemy wykorzystujące techniki uczenia głębokiego oparte o architekturę enkoder-dekoder (rys. 3.1), gdzie enkoder dokonuje ekstrakcji cech i mapuje je na przestrzeń utajoną na podstawie której dekoder dokonuje rekonstrukcji obiektu. Obie sieci zostały wyposażone w moduł enkodujący należący do tej samej rodziny konwolucyjnych sieci neuronowych, natomiast zastosowane moduły dekodujące zupełnie się różnią z uwagi na odmienne reprezentacje obiektów wyjściowych, które wymagają wykorzystania innych mechanizmów do odtwarzania obiektów. Pierwsze zaprezentowane podejście służy dekodowania obiektów w reprezentacji siatek wokseli, natomiast druga w postaci siatek wielokątów.

Content:
1. System rekonstruujący obiekty 3D w reprezentacji voxelowej <br>
  1.1 efekty <br>
  1.2 jak to działa
2. System rekonstruujący obiekty 3D w reprezentacji powierzchniowej <br>
  2.1 efekty <br>
  2.2 jak to działa
  
## 1. System rekonstruujący obiekty 3D w reprezentacji voxelowej
  
#### 1.1 efekty

<img src=https://user-images.githubusercontent.com/101074920/157133592-ad59ec54-3c23-49e5-bf37-e5797efa3064.png width="500">

<img src=https://user-images.githubusercontent.com/101074920/157133608-17edc289-1624-40e3-b1b6-65ef9f2ab046.png width="500">

<img src=https://user-images.githubusercontent.com/101074920/157133624-9a620aae-3442-40cc-a0cc-a56c5139b9bd.png width="500">

#### 1.2 jak to działa

<img src=https://user-images.githubusercontent.com/101074920/157141793-d0e579f9-c80f-426b-ad40-dc5379100366.png width="600">

<img src=https://user-images.githubusercontent.com/101074920/157141535-ae1263c9-1884-4edb-acb6-6e41b87453e1.png width="300">

## 2. System rekonstruujący obiekty 3D w reprezentacji powierzchniowej

#### 2.1 efekty

<img src=https://user-images.githubusercontent.com/101074920/157141560-c57aa8ff-76e8-40e6-ac48-e8892ac72c6e.png width="500">

<img src=https://user-images.githubusercontent.com/101074920/157141567-46602610-c6e9-4262-8067-705a6dcc74fb.png width="500">

<img src=https://user-images.githubusercontent.com/101074920/157143676-744753f2-5515-4de0-b898-af2515d59d69.png width="500">

<img src=https://user-images.githubusercontent.com/101074920/157141576-6fb3f904-2b19-4a28-9900-963dee2a0bee.png width="500">

#### 2.2 jak to działa

<img src=https://user-images.githubusercontent.com/101074920/157141626-a0eadc00-7ab6-47b0-b71a-976952987560.png width="600">

<img src=https://user-images.githubusercontent.com/101074920/157141631-8a80166a-c636-4312-a271-daa62b3d4ab2.png width="300">
