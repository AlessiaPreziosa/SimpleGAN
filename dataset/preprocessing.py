from utilities.dataset import *

'This script is used to filter outfits that present a top, a bottom, a pair of shoes and at least an accessory.'
'2 csv are created: the first containing the filtered clothes, the second containing the filtered outfits.'
'The second one will be used to split the dataset'

# returns JSON object as a dictionary
dataset = json.load(open(f'{root_dir}polyvore_outfits/polyvore_item_metadata.json'))  # 251008 clothes

# gather elements belonging to the following four categories: tops, bottoms, shoes, accessories (+jewelry);
# others are dropped
df = pd.DataFrame([[key, dataset[key]['semantic_category']] for key in dataset.keys()
                  if dataset[key]['semantic_category'] in ['tops', 'bottoms', 'shoes', 'accessories', 'jewellery']],
                  columns=['ID', 'Semantic Category'])  # 153905 clothes

# save actual used clothes in a directory that needs to be created at the path: root_dir/my_data
path = f'{root_dir}my_data/df.csv'
df.to_csv(path, index=False)

# divide ids of tops, bottoms, shoes and accessories
ids = list(df['ID'])
tops = [ids[i] for i in range(len(df)) if df.iloc[i]['Semantic Category'] == 'tops']   # 32998
bottoms = [ids[i] for i in range(len(df)) if df.iloc[i]['Semantic Category'] == 'bottoms']   # 27670
shoes = [ids[i] for i in range(len(df)) if df.iloc[i]['Semantic Category'] == 'shoes']    # 44850
accessories = [ids[i] for i in range(len(df)) if df.iloc[i]['Semantic Category']
               in ['accessories', 'jewellery']]  # 48387

# load all 68306 outfits in a dictionary to exclude duplicates
outfit = {}
for directory in ['disjoint', 'nondisjoint']:
    for file in ['train', 'test', 'valid']:
        outfit.update(make_dictionary(directory, file))

# filter only outfits presenting four items: tops, bottoms, shoes, accessories and save the dataset in a csv
real_outfit = pd.DataFrame(np.zeros((len(outfit), 4)))
j, k = 0, 0

for key in outfit.keys():
    ID = outfit[key]
    for i in range(len(ID)):
        if ID[i] in tops:
            real_outfit.iloc[j][k] = int(ID[i])
        elif ID[i] in bottoms:
            real_outfit.iloc[j][k + 1] = int(ID[i])
        elif ID[i] in shoes:
            real_outfit.iloc[j][k + 2] = int(ID[i])
        elif ID[i] in accessories:
            real_outfit.iloc[j][k + 3] = int(ID[i])
    if np.count_nonzero(np.array(real_outfit.iloc[j])) < 4:
        real_outfit.drop(real_outfit.index[j], inplace=True)
    else:
        j += 1

real_outfit.reset_index(drop=True, inplace=True)
real_outfit.columns = ['tops', 'bottoms', 'shoes', 'accessories']
path = f'{root_dir}my_data/real_outfit.csv'  # this is needed to split the set in training/validation/test
real_outfit.to_csv(path, index=False)  # 21774
