import sys
import gradio as gr

from models.cn_inpaint import ControlNetInpaint
from settings.prompts import ROOM_TYPES

def launch_cn_inpaint(host, port, debug=True, enable_queue=True):
    # Initialise model
    model = ControlNetInpaint()

    # Launch gradio app
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Tab("ControlNet Inpaint"):
                interface(model)
    app.launch(
        debug=debug,
        enable_queue=enable_queue,
        server_port=port,
        server_name=host
    )

def interface(model):
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                # Input image
                input_image = gr.Image(
                        source="upload",
                        tool="sketch",
                        elem_id="image_upload",
                        type="pil",
                        label="Upload",
                ).style(height=260)

                # Allow users to specify styles
                roomtype = (
                    gr.Dropdown(
                        choices=list(ROOM_TYPES.keys()),
                        value=list(ROOM_TYPES.keys())[0],
                        label="Room Type",
                    )
                )
                architecture_style = (
                    gr.Dropdown(
                        choices=ROOM_TYPES[roomtype.value]['architecture_style'],
                        value=ROOM_TYPES[roomtype.value]['architecture_style'][0],
                        label="Architecture Style",
                    )
                )

                # Allow users to specify optional prompts
                neg_prompt = gr.Textbox(
                    lines=1,
                    placeholder="Negative Prompt",
                    show_label=False
                )
                override_prompt = gr.Textbox(
                    lines=1,
                    placeholder="Override Prompt",
                    show_label=False
                )

                # Advanced options - more fine-grained control
                with gr.Accordion('Advance Options', open=False):
                    with gr.Row(), \
                         gr.Column():

                        guidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=12.4,
                                label="Guidance Scale",
                        )
                        strength_min = gr.Slider(
                            minimum=0.1,
                            maximum=1,
                            step=0.1,
                            value=0.1,
                            label="Min Control Strength",
                        )
                        strength_max = gr.Slider(
                                minimum=0.1,
                                maximum=1,
                                value=0.5,
                                label='Max Control Strength'
                        )
                        use_fixed_strength = gr.Checkbox(label="Use fixed strength")
                        num_steps = (
                            gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=20,
                                label="Num Inference Step",
                            )
                        )
                    with gr.Row(), \
                         gr.Column():

                        num_samples = gr.Slider(
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=5,
                                label="Number Of Images",
                        )
                        seed = gr.Slider(
                                minimum=0,
                                maximum=1000000,
                                step=1,
                                value=0,
                                label="Seed (0 for random)",
                        )
                        upscale = gr.Checkbox(label="Upscale")
                        mask_option = gr.Radio(["dilate", "erode"], value='dilate')
                        mask_dilation = gr.Slider(
                                minimum=0,
                                maximum=20,
                                step=1,
                                value=0,
                                label="Dilate/Erode Iteration",
                        )
                predict_button = gr.Button(value="Generate")
                mask_button = gr.Button(value="Generate Mask")


            # Show generated mask
            with gr.Column():
                mask_image = gr.Gallery(
                    label="Masked image",
                    show_label=False,
                    elem_id="gallery",
                )

            # Show outputs
            with gr.Column():
                output_image = gr.Gallery(
                    label="Generated images",
                    show_label=False,
                    elem_id="gallery",
                ).style(grid=(1, 2))

        # Dynamically change architecture_style based on room types
        def on_select(evt):
            return gr.update(
                    choices=ROOM_TYPES[evt]['architecture_style'],
                    value=ROOM_TYPES[evt]['architecture_style'][0]
            )
        roomtype.change(on_select, roomtype, architecture_style)

        # Generate
        predict_button.click(
            fn=model,
            inputs=[
                input_image,
                roomtype,
                architecture_style,
                neg_prompt,
                num_samples,
                guidance_scale,
                num_steps,
                strength_min,
                strength_max,
                seed,
                override_prompt,
                upscale,
                mask_dilation,
                mask_option,
                use_fixed_strength
            ],
            outputs=[output_image, mask_image],
        )

        mask_button.click(
            fn=model.generate_mask,
            inputs=[
                input_image,
                mask_dilation,
                mask_option
            ],
            outputs=[mask_image]

        )

if __name__=='__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3091
    launch_cn_inpaint('0.0.0.0', port)
