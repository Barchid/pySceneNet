# Add additional class lists here and select which one to use in
# the main method below.  The class list order is important, the
# script below gives 'Unknown' an id of 0, 'Bed' and id of 1 etc.
NYU_13_CLASSES = ['Unknown', 'Bed', 'Books', 'Ceiling', 'Chair',
                  'Floor', 'Furniture', 'Objects', 'Picture',
                  'Sofa', 'Table', 'TV', 'Wall', 'Window'
                  ]

NYU_40_CLASSES = [
    "Wall",
    "Floor",
    "Cabinet",
    "Bed",
    "Chair",
    "Sofa",
    "Table",
    "Door",
    "Window",
    "BookShelf",
    "Picture",
    "Counter",
    "Blinds",
    "Desks",
    "Shelves",
    "Curtain",
    "Dresser",
    "Pillow",
    "Mirror",
    "Floor-mat",
    "Clothes",
    "Ceiling",
    "Books",
    "Refrigerator",
    "Television",
    "Paper",
    "Towel",
    "Shower-curtain",
    "Box",
    "Whiteboard",
    "Person",
    "NightStand",
    "Toilet",
    "Sink",
    "Lamp",
    "Bathtub",
    "Bag",
    "Other-structure",
    "Other-furniture",
    "Other-prop"
]


if __name__ == '__main__':
    # Change these two variables if new columns are added to the wnid_to_class.txt file
    # This is the name of the column in the textfile
    column_name = '40_classes'
    # This is the list of classes, with the index in the list denoting the # class_id
    class_list = NYU_40_CLASSES

    wnid_to_classid = {}
    with open('wnid_to_class.txt', 'r') as f:
        class_lines = f.readlines()
        column_headings = class_lines[0].split()
        for class_line in class_lines[1:]:
            wnid = class_line.split()[0].zfill(8)
            classid = class_list.index(
                class_line.split()[column_headings.index(column_name)])
            wnid_to_classid[wnid] = classid
    print(wnid_to_classid)
