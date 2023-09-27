# SimpleGAN
The objective of this Deep Learning project will be to develop a Generative Adversarial Network (GAN) model to accomplish the outfit generation task in the fashion domain. The primary objective is to generate a set of three images: bottoms (trousers/skirt), shoes, and accessories, given an input image of a top (shirt/t-shirt).

## HOW TO
1. Make sure requirements from *requirements.txt* are installed: try **pip install -r requirements.txt** if download is not automatic 
2. Download the Polyvore Dataset from https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/view
3. Change the variable *root_dir* in *utilities/dataset.py* so that it points to the directory where the Polyvore Outfit Dataset is (e.g.: /Users/alessiapreziosa/Downloads/)
4. Create the directory *my_data* in *root_dir*
5. Run ***dataset/preprocessing.py***
6. Create the directory tree in *root_dir* as in: 

    root_dir/outfits/name_of_set/name_of_clothes
    for name_of_set in ['train','val','test'] and for name_of_clothes in ['tops','bottoms','shoes','accessories']
    
7. Run ***dataset/split.py***
8. Now, the dataset is ready!
9. Run ***main.py***
10. Now, **checkpoints** will be saved in models directory
11. Run ***test.py*** with saved checkpoints

## What's inside?
* **dataset directory**: scripts for preprocessing the dataset
* **models directory**: checkpoints of cGANs saved after running ***main.py***
* **utilities directory**: some utilities for preprocessing and training
* **GAN directory**: scripts for the GAN architecture
* **results directory**: images generated from testing (.png) and files generated from data preprocessing (.csv)
