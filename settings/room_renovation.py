import random

import settings.prompts as prompts

CONFIG = {
        'image_resolution': 512,
        'detect_resolution': 512,
        'ddim_steps': 20,
        'guess_mode': False,
        'control_strength': lambda: random.uniform(0.3,0.6),
        'guidance_scale': lambda: random.uniform(7,13),
        'seed': -1,
        'eta': 0.,
        'models': ['seg2image', 'depth2image']
}

def create_prompts(
        room_type,
        room_objects,
        architecture_style=None,
        color_theme=None,
        floor=None
):
    add_prompt = '. '.join([f'{p.capitalize()}' for p in prompts.ADD_PROMPT]) + '. '
    negative_prompt = '. '.join([f'{p.capitalize()}' for p in prompts.NEGATIVE_PROMPTS]) + '. '

    room_dict = prompts.ROOM_TYPES[room_type]

    # Room objects
    room_objects = set(room_objects + room_dict['room_objects'])
    room_objects = [m for r in room_objects
                    if len(m := prompts.OBJECT_MAPPINGS.get(r,r)) > 0]
    room_objects = ', '.join(random.sample(room_objects, 5))

    # Prominent objects
    prominent_objects = room_dict['prominent_objects']
    prominent_objects = [m for r in prominent_objects
                         if len(m := prompts.OBJECT_MAPPINGS.get(r,r)) > 0]
    prominent_objects = ', '.join(prominent_objects)

    # Floor
    if floor is None:
        floor = random.choice(room_dict['floors'])

    # Architecture style
    if architecture_style is None:
        architecture_style = random.choice(room_dict['architecture_style'])

    # Colors
    if color_theme is None:
        color_theme = random.choice(room_dict['color_themes'])

    # Random keywords
    random_phrase = random.choice(prompts.RANDOM_PHRASES)

    prompt = (
        f'This is a photograph of a {room_type}. '
        f'{architecture_style.capitalize()} interior design. '
        f'{color_theme.capitalize()} color theme. '
        f'{floor.capitalize()} flooring. '
        f'Includes: {prominent_objects}, {room_objects}. '
        f'{random_phrase.capitalize()}.'
    )
    return prompt, add_prompt, negative_prompt
