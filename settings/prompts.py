ADD_PROMPT = [
        'best quality'
]

NEGATIVE_PROMPTS = [
        "window",
        'door',
        'out of order',
        'deformed',
        'disfigured',
        "jpeg artifacts", 
        'longbody',
        'lowres',
        'bad anatomy',
        'bad hands',
        'missing fingers',
        'extra digit',
        'fewer digits',
        'cropped',
        'worst quality',
        'low quality',
        'rug'
]

RANDOM_PHRASES = [
        'High quality photography. Sharp, focused. Canon EOS R3',
        'Award winning photography. Nikon d5300. Photo from Pintrest'
]

ROOM_TYPES = {
        'bedroom':
            {
                'prominent_objects': [
                    'bed'
                ],
                'room_objects': [
                    'armchair',
                    'mirror',
                    'ottoman',
                    'table',
                    'cabinet',
                ],
                'floors': [
                    'warm maple hardwood',
                    'bamboo',
                    'laminate',
                ],
                'color_themes': [
                    'soft neutrals',
                    'warm neutrals',
                    'soothing space with soft pastels',
                    'muted pastels',
                    'bold and bright',
                    'earthy tones',
                    'shades of gray',
                    'rich jewel tones',
                ],
                'architecture_style': [
                    'coastal',
                    'rustic',
                    'modern',
                    'industrial',
                    'scandinavian',
                    'french provincial',
                ]
        },
        'living room': {
                'prominent_objects': [
                    'sofas',
                    'table',
                ],
                'room_objects': [
                    'television',
                    'chairs',
                    'cushion',
                    'rug',
                    'bookshelf',
                    'painting',
                    'lamp',
                    'light',
                    'sconce',
                    'curtains',
                    'side table',
                    'coffee table',
                    'armchair',
                ],
                'floors': [
                    'warm maple hardwood',
                    'tile',
                    'hardwood',
                ],
                'color_themes': [
                    'soft neutrals',
                    'warm neutrals',
                    'soothing space with soft pastels',
                    'muted pastels',
                    'bold',
                    'earthy tones',
                    'shades of gray',
                    'vintage',
                    'coastal',
                    'monochromatic',
                    'monochromatic',
                ],
                'architecture_style': [
                    'coastal',
                    'contemporary',
                    'modern',
                    'wooden',
                    'victorian',
                ]
        },
        'bathroom': {
                'prominent_objects': [
                    'sink',
                    'toilet'
                ],
                'room_objects': [
                    'shower',
                    'cupboards',
                    'bathtub',
                    'mirror',
                    'towel rack',
                    'bath mat',
                ],
                'floors': [
                    'warm maple hardwood'
                    'tile',
                    'wooden',
                    'concrete'
                ],
                'color_themes': [
                    'white and clean',
                    'blue and aqua',
                    'earthly tones',
                    'grey and pastel',
                    'soothing space with soft pastels'
                ],
                'architecture_style': [
                    'modern',
                    'coastal',
                    'rustic',
                    'mediterranean',
                ]
        },
        'kitchen': {
                'prominent_objects': [
                    'oven',
                    'refrigerator'
                ],
                'room_objects': [
                    'stools',
                    'wooden cabinet',
                    'painting',
                    'microwave',
                    'countertops',
                    'dishwasher',
                    'toaster',
                    'trash can',
                ],
                'floors': [
                    'warm maple hardwood'
                    'tile',
                    'wooden',
                    'concrete'
                ],
                'color_themes': [
                    'white and bright',
                    'warm and cozy',
                    'soft neutrals',
                    'wooden',
                    'soothing space with soft pastels'
                ],
                'architecture_style': [
                    'traditional',
                    'coastal',
                    'contemporary',
                    'italian',
                    'mediterranean',
                ]
        },
        'backyard': {
                'prominent_objects': [
                    'circular table',
                    'chairs',
                    'sofa',
                    'outdoor dining'
                ],
                'room_objects': [
                    'coffee table',
                    'swings',
                    'grill oven',
                    'plants',
                    'comfy chairs',
                    'table',
                ],
                'floors': [
                    'grass',
                    'gravel',
                    'pavers',
                    'concrete',
                    'brick',
                    'pavers',
                    'decking',
                    'artificial turf',
                ],
                'color_themes': [
                    'earthy tones',
                    'warm and cozy',
                    'soft neutrals',
                    'rustic charm',
                    'balanced'
                ],
                "extra_objects": [
                    'outdoor fireplace,',
                    # 'small gazebo,',
                    'swimming pool,',
                    'outdoor cooking area,'
                ],
                'architecture_style': [
                    'modern',
                    'traditional',
                    'coastal',
                    'farmhouse',
                    'zen'
                ]
        }
}

OBJECT_MAPPINGS = {
        'rug': 'carpet',
        'wall': '',
        'ceiling': '',
        'floor': '',
        'kitchen island': '',
        # 'bed': '',
        'chest of drawers': 'cabinet',
}
