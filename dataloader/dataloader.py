import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles
import os
from typing import Callable, List, Optional, Union


# Inspired by gauche DataLoader
# https://github.com/leojklarner/gauche


class DreamLoader():
    """
    Loads and cleans up your data
    """

    def __init__(self):
        self.features = None
        self.labels = None
        self.datasets = {
            "leffingwell": {
                "features": ["IsomericSMILES"],
                "task_dim": 113,
                "task": "binary",
                "n_datapoints": 3522,
                # 113 labels, multilabel prediction
                "labels": ['alcoholic', 'aldehydic', 'alliaceous', 'almond', 'animal', 'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy', 'berry', 'black currant', 'brandy', 'bread', 'brothy', 'burnt', 'buttery', 'cabbage', 'camphoreous', 'caramellic', 'catty', 'chamomile', 'cheesy', 'cherry', 'chicken', 'chocolate', 'cinnamon', 'citrus', 'cocoa', 'coconut', 'coffee', 'cognac', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruity', 'garlic', 'gasoline', 'grape', 'grapefruit', 'grassy', 'green', 'hay', 'hazelnut', 'herbal', 'honey', 'horseradish', 'jasmine', 'ketonic', 'leafy', 'leathery', 'lemon', 'malty', 'meaty', 'medicinal', 'melon', 'metallic', 'milky', 'mint', 'mushroom', 'musk', 'musty', 'nutty', 'odorless', 'oily', 'onion', 'orange', 'orris', 'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn', 'potato', 'pungent', 'radish', 'ripe', 'roasted', 'rose', 'rum', 'savory', 'sharp', 'smoky', 'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweet', 'tea', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable', 'violet', 'warm', 'waxy', 'winey', 'woody'],
                "validate": True
            },
            "mayhew_2022": {
                "features": ["IsomericSMILES"],
                "task_dim": 1,
                "task": "binary",
                "n_datapoints": 1799,
                # 1 label, odor probability prediction
                "labels": ["is_odor"],
                "validate": True,
            },
            "competition_train":{
                "features": ["Dataset", "Mixture 1", "Mixture 2"],
                "labels": ["Experimental Values"],
                "validate": False # nan values in columns, broken
            },
            "competition_leaderboard":{
                "features": ["Dataset", "Mixture 1", "Mixture 2"],
                "labels": ["Experimental Values"],
                "validate": False # nan values in columns, broken
            },
            "competition_test":{
                "features": ["Dataset", "Mixture 1", "Mixture 2"],
                "labels": ["Experimental Values"],
                "validate": False # nan values in columns, broken
            },
            "gs-lf": {
                "features": ["nonStereoSMILES"],
                "task_dim": 138,
                "task": "binary",
                "n_datapoints": 4983,
                # 138 labels, multilabel prediction
                "labels": ['alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal', 'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy', 'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt', 'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery', 'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean', 'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling', 'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin', 'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy', 'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth', 'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery', 'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic', 'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty', 'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone', 'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn', 'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted', 'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy', 'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet', 'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable', 'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'],
                "validate": True
            },
            "abraham_2012":{
                "features": ["IsomericSMILES"],
                "task_dim": 1,
                "task": "regression",
                "n_datapoints": 268,
                # 1 label, regression
                "labels": ['Log (1/ODT)'],
                "validate": True
            },
            "arctander_1960":{
                "features": ["IsomericSMILES"],
                "task_dim": 76,
                "task": "binary",
                "n_datapoints": 2580,
                # 76 labels, multiclass prediction
                "labels": ['acid', 'aldehydic', 'almond', 'ambre', 'animal', 'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'berry', 'brandy', 'buttery', 'camphoraceous', 'caramelic', 'citrus', 'coco', 'coconut', 'creamy', 'earthy', 'ethereal', 'fatty', 'floral', 'fruity', 'gassy', 'geranium', 'grape', 'green', 'hay', 'herbal', 'honey', 'hyacinth', 'jasmin', 'leafy', 'leather', 'lilac', 'lily', 'medicinal', 'metallic', 'mimosa', 'minty', 'mossy', 'mushroom', 'musky', 'musty', 'narcissus', 'nutty', 'oily', 'orange', 'orange-blossom', 'orris', 'peach', 'pear', 'pepper', 'phenolic', 'pine', 'pineapple', 'plum', 'powdery', 'rooty', 'rose', 'sandalwood', 'smoky', 'sour', 'spicy', 'sulfuraceous', 'tarry', 'tea', 'tobacco', 'vanilla', 'vanillin', 'violet', 'waxy', 'winey', 'woody'],
                "validate": True
            },      
            "aromadb_odor":{
                "features": ["IsomericSMILES"],
                "task_dim": 1,
                "task": "binary",
                "n_datapoints": 869,
                # 1 label, binary prediction
                "labels": ['is_odor'],
                "validate": True
            },
            "aromadb_descriptor":{
                "features": ["IsomericSMILES"],
                "task_dim": 127,
                "task": "binary",
                "n_datapoints": 814,
                # 1 label, binary prediction
                "labels": ['acetic', 'acid', 'alcoholic', 'almond', 'ammonia', 'aniseed', 'apple', 'apricot', 'balsamic', 'banana', 'bergamot', 'berry', 'bitter', 'bland', 'blueberry', 'bread', 'burnt', 'burnt sugar', 'butter', 'butterscotch', 'cabbage', 'camomile', 'camphoraceous', 'caramellic', 'caraway', 'carnation', 'carrot', 'cheese', 'cherry', 'cinnamon', 'citrus', 'cloves', 'cocoa', 'coconut', 'coffee', 'corn', 'coumarin', 'creamy', 'curry', 'dairy', 'disagreeable', 'earthy', 'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruity', 'gardenia', 'garlic', 'gasoline', 'grape', 'grass', 'greasy', 'green', 'green tea', 'herbaceous', 'honey', 'jasmine', 'lavender', 'leafy', 'lemon', 'licorice', 'lily', 'maple', 'meaty', 'melon', 'menthol', 'milky', 'mint', 'minty tea', 'moldy', 'mushroom', 'musky', 'musty', 'nutty', 'oily', 'onion', 'orange', 'orange blossom', 'peach', 'peanut butter', 'pepper', 'petroleum', 'pine', 'pineapple', 'plum', 'popcorn', 'potato', 'pungent', 'rancid', 'raspberry', 'red berry', 'resinous', 'roasted', 'rose', 'salt', 'salty', 'sassafras', 'savory', 'seaweed', 'smoky', 'sour', 'spearmint', 'spicy', 'strawberry', 'sulfurous', 'sweet', 'tallow', 'tarry', 'tea', 'tenacious', 'terpineol', 'tobacco', 'turpentine', 'vanilla', 'vegatable', 'vegetable', 'vinegar', 'walnut', 'waxy', 'whisky', 'winey', 'wintergreen', 'woody'],
                "validate": True
            },
            "flavornet":{
                "features": ["IsomericSMILES"],
                "task_dim": 195,
                "task": "binary",
                "n_datapoints": 716,
                # 195 label, multiclass prediction
                "labels": ['acid', 'alcohol', 'alkaline', 'alkane', 'almond', 'almond shell', 'amine', 'anise', 'apple', 'apple peel', 'apple, rose', 'apricot', 'baked', 'balsamic', 'banana', 'basil', 'beet', 'biscuit', 'bitter', 'bitter almond', 'black currant', 'boiled vegetable', 'box tree', 'bread', 'broccoli', 'brown sugar', 'burnt', 'burnt sugar', 'butter', 'butterscotch', 'cabbage', 'camomile', 'camphor', 'caramel', 'caraway', 'cardboard', 'carrot', 'cat', 'celery', 'cheese', 'chemical', 'cinnamon', 'citrus', 'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cologne', 'cooked meat', 'cooked potato', 'cooked vegetable', 'coriander', 'cotton candy', 'coumarin', 'cream', 'crushed bug', 'cucumber', 'curry', 'dill', 'dust', 'earth', 'ester', 'ether', 'fat', 'fecal', 'fennel', 'fish', 'flower', 'foxy', 'fresh', 'fried', 'fruit', 'garlic', 'gasoline', 'geranium', 'grape', 'grapefruit', 'grass', 'green', 'green bean', 'green leaf', 'green pepper', 'green tea', 'hawthorne', 'hazelnut', 'herb', 'honey', 'horseradish', 'hot milk', 'hummus', 'hyacinth', 'jasmine', 'lactone', 'lavender', 'leaf', 'lemon', 'lettuce', 'licorice', 'lilac', 'lily', 'magnolia', 'malt', 'mandarin', 'maple', 'marshmallow', 'meat', 'meat broth', 'medicine', 'melon', 'menthol', 'metal', 'mildew', 'mint', 'mold', 'moss', 'mothball', 'muguet', 'mushroom', 'must', 'mustard', 'nut', 'nutmeg', 'oil', 'onion', 'orange', 'orange peel', 'orris', 'paint', 'paper', 'pea', 'peach', 'peanut butter', 'pear', 'pepper', 'peppermint', 'pesticide', 'phenol', 'pine', 'pineapple', 'plastic', 'plum', 'popcorn', 'potato', 'prune', 'pungent', 'putrid', 'rancid', 'raspberry', 'resin', 'roast', 'roast beef', 'roasted meat', 'roasted nut', 'rose', 'rubber', 'seaweed', 'sharp', 'smoke', 'soap', 'solvent', 'sour', 'soy', 'spearmint', 'spice', 'straw', 'strawberry', 'sulfur', 'sweat', 'sweet', 'tallow', 'tar', 'tart lime', 'tea', 'thiamin', 'thyme', 'tobacco', 'tomato', 'tomato leaf', 'truffle', 'turpentine', 'urine', 'vanilla', 'vinyl', 'violet', 'walnut', 'warm', 'watermelon', 'wax', 'wet cloth', 'whiskey', 'wine', 'wintergreen', 'wood', 'yeast',],
                "validate": True
            },
            "ifra_2019":{
                "features": ["IsomericSMILES"],
                "task_dim": 184,
                "task": "binary",
                "n_datapoints": 1146,
                # 184 label, multiclass prediction
                "labels": ['acidic', 'aldehydic', 'almond', 'amber', 'animal like', 'anisic', 'apple', 'apricot', 'aromatic', 'artemisia', 'balsamic', 'banana', 'bayleaf', 'bell pepper', 'bergamot', 'berry', 'bitter', 'blackcurrant', 'blueberry', 'brandy', 'burnt', 'butterscotch', 'buttery', 'camphoraceous', 'caramel', 'carrot', 'cedarwood', 'celery', 'chamomile', 'cheesy', 'cherry', 'cherry-blossom', 'chocolate', 'cinnamon', 'citronella', 'citrus', 'clean', 'clove', 'cocoa', 'coconut', 'coffee', 'cooling', 'coriander', 'corn', 'coumarin', 'creamy', 'cucumber', 'cumin', 'cyclamen', 'dairy', 'dry', 'earthy', 'ethereal', 'eucalyptus', 'fatty', 'fermented', 'floral', 'foliage', 'food like', 'fresh', 'fruity', 'fungal', 'galbanum', 'gardenia', 'garlic', 'geranium', 'gourmand', 'grape', 'grapefruit', 'grassy', 'green', 'hawthorn', 'hay', 'herbal', 'honey', 'honeydew', 'hyacinth', 'indolic', 'jasmin', 'juicy', 'juniper', 'kiwi', 'lactonic', 'lavender', 'leafy', 'leathery', 'lemon', 'lemongrass', 'licorice', 'light', 'lily', 'lime', 'linden', 'magnolia', 'mandarin', 'mango', 'maple', 'marigold', 'marine', 'medicinal', 'melon', 'menthol', 'metallic', 'milky', 'minty', 'mossy', 'muguet', 'mushroom', 'musk like', 'musty', 'narcissus', 'nasturtium', 'neroli', 'neutral', 'nutmeg', 'nutty', 'oak', 'oily', 'onion', 'orange', 'orris', 'osmanthus', 'ozonic', 'patchouli', 'peach', 'peanut', 'pear', 'peppermint', 'peppery', 'petitgrain', 'phenolic', 'pine', 'pineapple', 'plastic', 'plum', 'popcorn', 'potato', 'powdery', 'powerful', 'pungent', 'raspberry', 'rhubarb', 'roasted', 'rooty', 'rose', 'rosemary', 'rubbery', 'rum', 'saffron', 'sage', 'sandalwood', 'savoury', 'sharp', 'smoky', 'soapy', 'soft', 'sour', 'spearmint', 'spicy', 'strawberry', 'sulfurous', 'sweet', 'sweet-pea', 'tangerine', 'tarragon', 'tea', 'terpenic', 'tobacco', 'tobacco like', 'tonka', 'tropical-fruit', 'tuberose', 'vanilla', 'vegetable', 'verbena', 'vetiver', 'violet', 'warm', 'watermelon', 'watery', 'waxy', 'woody', 'yeast', 'ylang'],
                "validate": True
            },
            "sharma_2021a":{
                "features": ["IsomericSMILES"],
                "task_dim": 572,
                "task": "binary",
                "n_datapoints": 3997,
                # 572 label, multiclass prediction
                "labels": ['absinthe', 'acacia', 'acetic', 'acetoin', 'acetone', 'acidic', 'acorn', 'acrid', 'acrylate', 'acrylic', 'agarwood', 'agreeable', 'alcoholic', 'aldehydic', 'algae', 'alkane', 'alliaceous', 'allium', 'allspice', 'almond', 'amber', 'ambergris', 'ambrette', 'ammonical', 'angelica', 'animal', 'anisic', 'antiseptic', 'apple', 'apricot', 'aromatic', 'arrack', 'artichoke', 'asparagus', 'astringent', 'autumn', 'bacon', 'bad', 'baked', 'balsamic', 'banana', 'barley', 'basil', 'bay', 'beany', 'beefy', 'beer', 'beeswax', 'bell pepper', 'benzaldehyde', 'benzene', 'benzoin', 'bergamot', 'berry', 'biscuit', 'bitter', 'black', 'blackberry', 'bland', 'bloody', 'blossom', 'blue cheese', 'blueberry', 'boiled', 'borneol', 'boronia', 'bouillon', 'box tree', 'brandy', 'bready', 'broccoli', 'brothy', 'brown', 'bubblegum', 'buchu', 'bud', 'burnt', 'buttermilk', 'butterscotch', 'buttery', 'cabbage', 'calm', 'camomile', 'camphoraceous', 'cananga', 'candy', 'cantaloupe', 'capers', 'caprylic', 'caramel', 'caraway', 'carbide', 'cardamom', 'cardboard', 'carnation', 'carrot', 'carvone', 'cashew', 'cassia', 'castoreum', 'catty', 'cauliflower', 'cedar', 'celery', 'cereal', 'chamomile', 'characteristic', 'charred', 'cheese', 'chemical', 'cherry', 'chervil', 'chicken', 'chicory', 'chilli', 'chip', 'chlorine', 'chloroform', 'chocolate', 'choking', 'chrysanthemum', 'cider', 'cilantro', 'cinnamyl', 'cistus', 'citral', 'citronella', 'citrus', 'civet', 'clam', 'clary', 'clean', 'cloth', 'clove', 'clover', 'cloying', 'coal tar', 'cocoa', 'coconut', 'coffee', 'cognac', 'cologne', 'coniferous', 'cooked', 'cookie', 'cooling', 'coriander', 'corn', 'cortex', 'costus', 'cotton', 'coumarinic', 'cranberry', 'creamy', 'cress', 'crispy', 'cucumber', 'cumin', 'currant', 'curry', 'custard', 'cyclamen', 'cypress', 'dairy', 'damascone', 'dank', 'dark chocolate', 'date', 'davana', 'dead animal', 'decayed', 'decomposing', 'deep fried', 'delicate', 'dewy', 'diffusive', 'dihydrojasmone', 'dill', 'dirty', 'disagreeable', 'distinctive', 'dried', 'dry', 'durain', 'dusty', 'earthy', 'eggs', 'eggy', 'elderberry', 'elderflower', 'elemi', 'erogenic', 'estery', 'ethereal', 'eucalyptus', 'eugenol', 'farnesol', 'fatty', 'fecal', 'feet', 'fennel', 'fenugreek', 'fermented', 'fern', 'feta', 'fig', 'filbert', 'fir', 'fir cone', 'fir needle', 'fishy', 'fleshy', 'floral', 'foliage', 'forest', 'foul', 'foxy', 'fragrant', 'frankincense', 'freesia', 'fresh', 'fried', 'fruity', 'fuel', 'fungal', 'furfural', 'fusel', 'galbanum', 'gamey', 'garbage', 'gardenia', 'garlic', 'gasoline', 'gassy', 'genet', 'geranium', 'ginger', 'glue', 'goaty', 'gooseberry', 'gourmand', 'graham cracker', 'grain', 'grape', 'grapefruit', 'grassy', 'gravy', 'greasy', 'green', 'green pea', 'green peach', 'green pear', 'grilled', 'groundnut', 'guaiacol', 'guaiacwood', 'guava', 'hairy', 'ham', 'harsh', 'hawthorn', 'hay', 'hazelnut', 'heliotrope', 'herbaceous', 'high', 'honey', 'honeydew', 'honeysuckle', 'horseradish', 'humus', 'hyacinth', 'hydrocarbon', 'immortelle', 'incense', 'indole', 'insipid', 'irritating', 'isojasmone', 'jackfruit', 'jammy', 'jasmine', 'jasmone', 'juicy', 'juniper', 'ketonic', 'kimchi', 'kiwi', 'labdanum', 'lactonic', 'lamb', 'lard', 'laundered', 'lavender', 'leafy', 'leathery', 'leek', 'lemon', 'lemongrass', 'lettuce', 'licorice', 'lie', 'lilac', 'lily', 'lily of the valley', 'limburger', 'lime', 'linalool', 'linalyl', 'linden', 'linen', 'linseed', 'loganberry', 'longifolene', 'lovage', 'lychee', 'mace', 'magnolia', 'mahogany', 'malty', 'mandarin', 'mango', 'manure', 'maple', 'maraschino', 'marine', 'marjoram', 'marshmallow', 'marshy', 'marzipan', 'meaty', 'medicinal', 'melon', 'mentholic', 'mercaptan', 'metallic', 'mildew', 'milky', 'mimosa', 'minty', 'molasses', 'moldy', 'mossy', 'mothball', 'mousy', 'muguet', 'mushroom', 'musk', 'mustard', 'musty', 'mutton', 'myrrh', 'nail polish', 'naphthyl', 'narcissus', 'nasturtium', 'natural', 'nauseating', 'neroli', 'new mown hay', 'nitrile', 'nutmeg', 'nutty', 'oakmoss', 'obnoxious', 'ocean', 'odorless', 'offensive', 'oily', 'old wood', 'onion', 'opoponax', 'orange', 'orchid', 'organic', 'oriental', 'origanum', 'orris', 'osmanthus', 'oyster', 'ozone', 'paint', 'palmarosa', 'papaya', 'paper', 'parmesan', 'parsley', 'passion fruit', 'patchouli', 'peach', 'peanut', 'peanut butter', 'pear', 'peely', 'peony', 'pepper', 'peppermint', 'persistent', 'pesticide', 'petal', 'petitgrain', 'petroleum', 'phenolic', 'pine', 'pineapple', 'piperidine', 'pistachio', 'plastic', 'pleasant', 'plowed', 'plum', 'pollen', 'pomegranate', 'popcorn', 'pork', 'potato', 'poultry', 'powdery', 'praline', 'privet', 'prune', 'pulpy', 'pumpkin', 'pungent', 'putrid', 'pyrazine', 'pyridine', 'quince', 'quinoline', 'radish', 'raisin', 'rancid', 'raspberry', 'raw', 'refreshing', 'repellent', 'repulsive', 'reseda', 'resinous', 'rhubarb', 'rich', 'ripe', 'roasted', 'root', 'root beer', 'rooty', 'roquefort', 'rose', 'rosemary', 'rosewood', 'rotten', 'rotten fish', 'rotten vegetables', 'rotting fruit', 'rubbery', 'rue', 'rummy', 'rye', 'saffron', 'sage', 'salmon', 'salty', 'sandalwood', 'sandy', 'sappy', 'sarsaparilla', 'sassafras', 'sauerkraut', 'sausage', 'savory', 'sawdust', 'scallion', 'scotch', 'seafood', 'seashore', 'seaweed', 'seedy', 'sewage', 'shellfish', 'shrimp', 'sickening', 'skunk', 'smoky', 'soapy', 'soil', 'solvent', 'soupy', 'sour', 'soy', 'spearmint', 'sperm', 'spicy', 'spruce', 'starfruit', 'stench', 'stinging', 'stinky', 'storax', 'strawberry', 'styrene', 'sugar', 'sulfurous', 'swampy', 'sweaty', 'sweet', 'sweet pea', 'swimming pool', 'taco', 'tagette', 'tallow', 'tangerine', 'tarragon', 'tarry', 'tart', 'tea', 'tea rose', 'tequila', 'terpenic', 'thiamine', 'thujonic', 'thyme', 'tiglate', 'toasted', 'tobacco', 'toffee', 'tolu', 'tomato', 'tonka', 'tropical', 'truffle', 'tuberose', 'turkey', 'turnip', 'turpentine', 'tutti frutti', 'umami', 'unpleasant', 'unripe banana', 'unripe fruit', 'urine', 'valerian', 'vanilla', 'varnish', 'vegetable', 'velvet', 'verbena', 'vetiver', 'vine', 'vinegar', 'vinyl', 'violet', 'violet leaf', 'violet leafy', 'walnut', 'warm', 'wasabi', 'watercress', 'watermelon', 'watery', 'wax', 'weedy', 'wet', 'whiskey', 'wild', 'winey', 'wintergreen', 'woody', 'worty', 'yeasty', 'ylang', 'yogurt', 'zesty'],
                "validate": True
            },
            "sigma_2014":{
                "features": ["IsomericSMILES"],
                "task_dim": 116,
                "task": "binary",
                "n_datapoints": 872,
                # 116 label, multiclass prediction
                "labels": ['alcohol', 'alliaceous', 'almond', 'animal', 'anise', 'apple', 'apricot', 'balsam', 'balsamic', 'banana', 'beef', 'beer', 'berry', 'blossom', 'blueberry', 'brandy', 'butter', 'butterscotch', 'cabbage', 'camphoraceous', 'cantaloupe', 'caramel', 'caraway', 'carnation', 'cedar', 'celery', 'cheese', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clove', 'coconut', 'coffee', 'corn', 'coumarin', 'cranberry', 'creamy', 'cucumber', 'earthy', 'ethereal', 'fatty', 'fennel', 'fishy', 'floral', 'fruity', 'gardenia', 'geranium', 'grape', 'grapefruit', 'green', 'hawthorne', 'hazelnut', 'herba-', 'herbaceous', 'honey', 'horseradish', 'hyacinth', 'iris', 'jam', 'jasmine', 'lavender', 'leafy', 'lemon', 'lilac', 'lily', 'lime', 'mango', 'maple', 'marigold', 'meaty', 'medicinal', 'melon', 'minty', 'mossy', 'mushroom', 'musty', 'nutty', 'oily', 'orange', 'peach', 'peanut', 'pear', 'pepper', 'pineapple', 'plastic', 'plum', 'potato', 'quince', 'raspberry', 'rich', 'rose', 'rum', 'sage', 'seedy', 'smoky', 'soapy', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweet', 'tart', 'tobacco', 'tropical', 'turpentine', 'vanilla', 'vegetable', 'violet', 'walnut', 'warm', 'waxy', 'whiskey', 'wine-like', 'winelike', 'woody'],
                "validate": True
            },
            "keller_2016":{
                "features": ["IsomericSMILES", "Dilution"],
                "task_dim": 110,
                "task": "regression",
                "n_datapoints": 960,
                # 110 label, regression
                "labels": ['Acid_mean', 'Acid_stdev', 'Acid_median', 'Acid_nonzero_mean', 'Acid_nonzero_proportion', 'Ammonia_mean', 'Ammonia_stdev', 'Ammonia_median', 'Ammonia_nonzero_mean', 'Ammonia_nonzero_proportion', 'Bakery_mean', 'Bakery_stdev', 'Bakery_median', 'Bakery_nonzero_mean', 'Bakery_nonzero_proportion', 'Burnt_mean', 'Burnt_stdev', 'Burnt_median', 'Burnt_nonzero_mean', 'Burnt_nonzero_proportion', 'Chemical_mean', 'Chemical_stdev', 'Chemical_median', 'Chemical_nonzero_mean', 'Chemical_nonzero_proportion', 'Cold_mean', 'Cold_stdev', 'Cold_median', 'Cold_nonzero_mean', 'Cold_nonzero_proportion', 'Decayed_mean', 'Decayed_stdev', 'Decayed_median', 'Decayed_nonzero_mean', 'Decayed_nonzero_proportion', 'Familiarity_mean', 'Familiarity_stdev', 'Familiarity_median', 'Familiarity_nonzero_mean', 'Familiarity_nonzero_proportion', 'Fish_mean', 'Fish_stdev', 'Fish_median', 'Fish_nonzero_mean', 'Fish_nonzero_proportion', 'Flower_mean', 'Flower_stdev', 'Flower_median', 'Flower_nonzero_mean', 'Flower_nonzero_proportion', 'Fruit_mean', 'Fruit_stdev', 'Fruit_median', 'Fruit_nonzero_mean', 'Fruit_nonzero_proportion', 'Garlic_mean', 'Garlic_stdev', 'Garlic_median', 'Garlic_nonzero_mean', 'Garlic_nonzero_proportion', 'Grass_mean', 'Grass_stdev', 'Grass_median', 'Grass_nonzero_mean', 'Grass_nonzero_proportion', 'Intensity_mean', 'Intensity_stdev', 'Intensity_median', 'Intensity_nonzero_mean', 'Intensity_nonzero_proportion', 'Musky_mean', 'Musky_stdev', 'Musky_median', 'Musky_nonzero_mean', 'Musky_nonzero_proportion', 'Pleasantness_mean', 'Pleasantness_stdev', 'Pleasantness_median', 'Pleasantness_nonzero_mean', 'Pleasantness_nonzero_proportion', 'Sour_mean', 'Sour_stdev', 'Sour_median', 'Sour_nonzero_mean', 'Sour_nonzero_proportion', 'Spices_mean', 'Spices_stdev', 'Spices_median', 'Spices_nonzero_mean', 'Spices_nonzero_proportion', 'Sweaty_mean', 'Sweaty_stdev', 'Sweaty_median', 'Sweaty_nonzero_mean', 'Sweaty_nonzero_proportion', 'Sweet_mean', 'Sweet_stdev', 'Sweet_median', 'Sweet_nonzero_mean', 'Sweet_nonzero_proportion', 'Warm_mean', 'Warm_stdev', 'Warm_median', 'Warm_nonzero_mean', 'Warm_nonzero_proportion', 'Wood_mean', 'Wood_stdev', 'Wood_median', 'Wood_nonzero_mean', 'Wood_nonzero_proportion'],
                "validate": True
            },
        }

    def get_dataset_names(self, valid_only: Optional[bool] = True) -> List[str]:
        names = []
        if valid_only:
            for k, v in self.datasets.items():
                if v['validate']:
                    names.append(k)
        else:
            names = list(self.datasets.keys())
        return names
    
    def get_dataset_specifications(self, name: str) -> dict:
        assert name in self.datasets.keys(), (
            f"The specified dataset choice ({name}) is not a valid option. "
            f"Choose one of {list(self.datasets.keys())}."
        )
        return self.datasets[name]

    def read_csv(self,
                 path: str,
                 smiles_column: str,
                 label_columns: List[str],
                 validate: bool = True,
                 ) -> None:
        """
        Loads a csv and stores it as features and labels.
        """
        assert isinstance(
            smiles_column, List
        ), f"smiles_column ({smiles_column}) must be a list of strings"
        assert isinstance(label_columns, list) and all(isinstance(item, str) for item in label_columns), "label_columns ({label_columns}) must be a list of strings."

        df = pd.read_csv(path, usecols=[*smiles_column, *label_columns])
        self.features = df[smiles_column].to_numpy()
        if len(smiles_column) == 1: 
            self.features = self.features.flatten()

        self.labels = df[label_columns].values
        if validate:
            self.validate()

    def load_benchmark(self,
                       name: str,
                       path=None,
                       validate: bool = True,
                       ) -> None:
        """
        Pulls existing benchmark from datasets.
        """
        assert name in self.datasets.keys(), (
            f"The specified dataset choice ({name}) is not a valid option. "
            f"Choose one of {list(self.datasets.keys())}."
        )

        # if no path is specified, use the default data directory
        if path is None:
            path = os.path.abspath(
                os.path.join(
                    os.path.abspath(__file__),
                    "..",
                    "..",
                    "datasets",
                    name,
                    name + "_combined.csv",
                )
            )

        self.read_csv(
            path=path,
            smiles_column=self.datasets[name]["features"],
            label_columns=self.datasets[name]["labels"],
            validate=self.datasets[name]["validate"],
        )        

        if not self.datasets[name]["validate"]:
            print(f"{name} dataset is known to have invalid entries. Validation is turned off.")

    def validate(self, 
                 drop: Optional[bool] = True, 
                 canonicalize: Optional[bool] = True
    ) -> None:
        """
        Utility function to validate a read-in dataset of smiles and labels by
        checking that all SMILES strings can be converted to rdkit molecules
        and that all labels are numeric and not NaNs.
        Optionally drops all invalid entries and makes the
        remaining SMILES strings canonical (default).

        :param drop: whether to drop invalid entries
        :type drop: bool
        :param canonicalize: whether to make the SMILES strings canonical
        :type canonicalize: bool
        """
        invalid_mols = np.array(
            [
                True if MolFromSmiles(x) is None else False
                for x in self.features
            ]
        )
        if np.any(invalid_mols):
            print(
                f"Found {invalid_mols.sum()} SMILES strings "
                f"{[x for i, x in enumerate(self.features) if invalid_mols[i]]} "
                f"at indices {np.where(invalid_mols)[0].tolist()}"
            )
            print(
                "To turn validation off, use dataloader.read_csv(..., validate=False)."
            )

        invalid_labels = np.isnan(self.labels).squeeze()
        if np.any(invalid_labels):
            print(
                f"Found {invalid_labels.sum()} invalid labels "
                f"{self.labels[invalid_labels].squeeze()} "
                f"at indices {np.where(invalid_labels)[0].tolist()}"
            )
            print(
                "To turn validation off, use dataloader.read_csv(..., validate=False)."
            )
        if invalid_labels.ndim > 1:
            invalid_idx = np.any(np.hstack((invalid_mols.reshape(-1, 1), invalid_labels)), axis=1)
        else:
            invalid_idx = np.logical_or(invalid_mols, invalid_labels)

        if drop:
            self.features = [
                x for i, x in enumerate(self.features) if not invalid_idx[i]
            ]
            self.labels = self.labels[~invalid_idx]
            assert len(self.features) == len(self.labels)

        if canonicalize:
            self.features = [
                MolToSmiles(MolFromSmiles(smiles), isomericSmiles=False)
                for smiles in self.features
            ]

    def featurize(
        self, representation: Union[str, Callable], **kwargs
    ) -> None:
        """Transforms SMILES into the specified molecular representation.

        :param representation: the desired molecular representation.
        :type representation: str or Callable
        :param kwargs: additional keyword arguments for the representation function
        :type kwargs: dict
        """

        assert isinstance(representation, (str, Callable)), (
            f"The specified representation choice {representation} is not "
            f"a valid type. Please choose a string from the list of available "
            f"representations or provide a callable that takes a list of "
            f"SMILES strings as input and returns the desired featurization."
        )

        valid_representations = [
            "graphein_molecular_graphs",
            "pyg_molecular_graphs",
            "molecular_graphs",
            "morgan_fingerprints",
            "rdkit2d_normalized_features",
            "mordred_descriptors",
            "competition_smiles",
            "competition_rdkit2d"
        ]

        if isinstance(representation, Callable):
            self.features = representation(self.features, **kwargs)

        elif representation == "graphein_molecular_graphs":
            from .representations.graphs import graphein_molecular_graphs

            self.features = graphein_molecular_graphs(smiles=self.features, **kwargs)

        elif representation == "pyg_molecular_graphs":
            from .representations.graphs import pyg_molecular_graphs

            self.features = pyg_molecular_graphs(smiles=self.features, **kwargs)

        elif representation == "molecular_graphs":
            from .representations.graphs import molecular_graphs

            self.features = molecular_graphs(smiles=self.features, **kwargs)

        elif representation == "morgan_fingerprints":
            from .representations.features import morgan_fingerprints

            self.features = morgan_fingerprints(self.features, **kwargs)

        elif representation == "rdkit2d_normalized_features":
            from .representations.features import rdkit2d_normalized_features

            self.features = rdkit2d_normalized_features(self.features, **kwargs)

        elif representation == "mordred_descriptors":
            from .representations.features import mordred_descriptors

            self.features = mordred_descriptors(self.features, **kwargs)

        elif representation == "competition_smiles":
            # Features is ["Dataset", "Mixture 1", "Mixture 2"]
            smi_df = pd.read_csv("/Users/ellarajaonson/Documents/dream/dreamloader/datasets/competition_train/mixture_smi_definitions_clean.csv")
            feature_list = []
            for feature in self.features:
                mix_1 = smi_df.loc[(smi_df['Dataset'] == feature[0]) & (smi_df['Mixture Label'] == feature[1])][smi_df.columns[2:]]
                mix_1 = mix_1.dropna(axis=1).to_numpy()[0]
                mix_2 = smi_df.loc[(smi_df['Dataset'] == feature[0]) & (smi_df['Mixture Label'] == feature[2])][smi_df.columns[2:]]
                mix_2 = mix_2.dropna(axis=1).to_numpy()[0]
                feature_list.append([mix_1, mix_2])

            self.features = np.array(feature_list, dtype=object)

        elif representation == "competition_rdkit2d":
            # Features is ["Dataset", "Mixture 1", "Mixture 2"]
            rdkit_df = pd.read_csv("datasets/competition_train/mixture_rdkit_definitions_clean.csv")
            feature_list = []
            for feature in self.features:
                mix_1 = rdkit_df.loc[(rdkit_df['Dataset'] == feature[0]) & (rdkit_df['Mixture Label'] == feature[1])][rdkit_df.columns[2:]]
                mix_1 = mix_1.dropna(axis=1).to_numpy()[0]
                mix_2 = rdkit_df.loc[(rdkit_df['Dataset'] == feature[0]) & (rdkit_df['Mixture Label'] == feature[2])][rdkit_df.columns[2:]]
                mix_2 = mix_2.dropna(axis=1).to_numpy()[0]
                feature_list.append([mix_1, mix_2])

            self.features = np.array(feature_list, dtype=object)

        elif representation == "competition_rdkit2d_augment":
            # Features is ["Dataset", "Mixture 1", "Mixture 2"]
            rdkit_df = pd.read_csv("datasets/competition_train/mixture_rdkit_definitions_clean.csv")
            feature_list = []
            feature_list_augment = []
            for feature in self.features:
                mix_1 = rdkit_df.loc[(rdkit_df['Dataset'] == feature[0]) & (rdkit_df['Mixture Label'] == feature[1])][rdkit_df.columns[2:]]
                mix_1 = mix_1.dropna(axis=1).to_numpy()[0]
                mix_2 = rdkit_df.loc[(rdkit_df['Dataset'] == feature[0]) & (rdkit_df['Mixture Label'] == feature[2])][rdkit_df.columns[2:]]
                mix_2 = mix_2.dropna(axis=1).to_numpy()[0]
                feature_list.append([mix_1, mix_2])
                feature_list_augment.append([mix_2, mix_1])
            feature_list += feature_list_augment

            self.features = np.array(feature_list, dtype=object)

        elif representation == "only_augment":

            feature_list_augment = np.array([[x[1], x[0]] for x in self.features])

            self.features = np.vstack((self.features, feature_list_augment))

        else:
            raise Exception(
                f"The specified representation choice {representation} is not a valid option."
                f"Choose between {valid_representations}."
            )
