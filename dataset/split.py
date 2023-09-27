# import libraries
from sklearn.model_selection import train_test_split
from utilities.dataset import *

new_root_dir = f'{root_dir}polyvore_outfits/images'

# load dataset reading the saved file
real_outfit = pd.read_csv(f'{root_dir}my_data/real_outfit.csv')

# split dataset in train(70%, 15241), validation(10%, 2178) and testing(20%, 4355)
train_outfit, test_outfit = train_test_split(real_outfit, test_size=0.2, random_state=42)
train_outfit, val_outfit = train_test_split(train_outfit, test_size=0.125, random_state=42)

# split train DataFrame in lists (train: tops, bottoms, shoes, accessories) and cast element to string
train_tops = cast(cast(list(train_outfit['tops']), int), str)
train_bottoms = cast(cast(list(train_outfit['bottoms']), int), str)
train_shoes = cast(cast(list(train_outfit['shoes']), int), str)
train_accessories = cast(cast(list(train_outfit['accessories']), int), str)

# split validation DataFrame in lists (validation: tops, bottoms, shoes, accessories) and cast element to string
val_tops = cast(cast(list(val_outfit['tops']), int), str)
val_bottoms = cast(cast(list(val_outfit['bottoms']), int), str)
val_shoes = cast(cast(list(val_outfit['shoes']), int), str)
val_accessories = cast(cast(list(val_outfit['accessories']), int), str)

# split test DataFrame in lists (test: tops, bottoms, shoes, accessories) and cast element to string
test_tops = cast(cast(list(test_outfit['tops']), int), str)
test_bottoms = cast(cast(list(test_outfit['bottoms']), int), str)
test_shoes = cast(cast(list(test_outfit['shoes']), int), str)
test_accessories = cast(cast(list(test_outfit['accessories']), int), str)

# folders root_dir/outfits/name_of_set/name_of_clothes
# for name_of_set in ['train','val','test'] and for name_of_clothes in ['tops','bottoms','shoes','accessories']
# need to be created

# save them in folders: root_dir/outfits/train/name_of_clothes
save(train_tops, new_root_dir, 'train', 'tops')
save(train_bottoms, new_root_dir, 'train', 'bottoms')
save(train_shoes, new_root_dir, 'train', 'shoes')
save(train_accessories, new_root_dir, 'train', 'accessories')

# save them in folders: root_dir/outfits/val/name_of_clothes
save(val_tops, new_root_dir, 'val', 'tops')
save(val_bottoms, new_root_dir, 'val', 'bottoms')
save(val_shoes, new_root_dir, 'val', 'shoes')
save(val_accessories, new_root_dir, 'val', 'accessories')

# save them in folders: root_dir/outfits/test/name_of_clothes
save(test_tops, new_root_dir, 'test', 'tops')
save(test_bottoms, new_root_dir, 'test', 'bottoms')
save(test_shoes, new_root_dir, 'test', 'shoes')
save(test_accessories, new_root_dir, 'test', 'accessories')




