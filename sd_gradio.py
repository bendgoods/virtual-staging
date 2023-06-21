import gradio as gr

from models.model import VirtualStagingModel
from settings.prompts import ROOM_TYPES


def interface(fn):
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    source="upload",
                    tool="sketch",
                    elem_id="image_upload",
                    type="pil",
                    label="Upload",
                ).style(height=260)

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


                with gr.Accordion('Advance Options', open=False):
                    with gr.Row():
                        with gr.Column():
                            gauidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=12.4,
                                label="Guidance Scale",
                            )

                            num_steps = (
                                gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=20,
                                    label="Num Inference Step",
                                )
                            )

                        with gr.Row():
                            with gr.Column():
                                num_samples = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    step=1,
                                    value=4,
                                    label="Number Of Images",
                                )
                                seeds = (
                                    gr.Slider(
                                        minimum=0,
                                        maximum=1000000,
                                        step=1,
                                        value=0,
                                        label="Seed(0 for random)",
                                    )
                                )

                predict_button = gr.Button(
                    value="Generate"
                )

            with gr.Column():
                mask_image = gr.Gallery(
                    label="Masked image",
                    show_label=False,
                    elem_id="gallery",
                )

            with gr.Column():
                output_image = gr.Gallery(
                    label="Generated images",
                    show_label=False,
                    elem_id="gallery",
                ).style(grid=(1, 2))



        # with gr.Column():
        #     gr.Examples(
        #         examples=[
        #             [
        #                 "sample_imgs/Empty_Room_1.jpg",
        #                 "a living room with mid century modern furniture",
        #             ],
        #             [
        #                 "sample_imgs/Empty_Room_2.jpg",
        #                 "dreamy sunken living room conversation pit, wooden floor, Bauhaus furniture and decoration, cozy atmosphere",
        #             ],
        #             [
        #                 "sample_imgs/Empty_Room_3.jpg",
        #                 "a sitting area with minimalist style furniture and a coffee table",
        #             ],
        #             [
        #                 "sample_imgs/Empty_Room_4.jpg",
        #                 "a cozy living room with warm and inviting furniture, a plush sofa and armchair upholstered in a rich, burgundy fabric and a wooden coffee table sits at the center of the room",
        #             ],

        #         ],
        #         inputs=[input_image, input_prompt],
        #         outputs=None,
        #         fn=None,
        #         cache_examples=False,
        #     )

        def on_select(evt):
            return gr.update(
                    choices=ROOM_TYPES[evt]['architecture_style'],
                    value=ROOM_TYPES[evt]['architecture_style'][0]) 

        roomtype.change(on_select, roomtype, architecture_style)



        predict_button.click(
            fn=fn,
            inputs=[
                input_image,
                roomtype,
                architecture_style,
                neg_prompt,
                num_samples,
                gauidance_scale,
                num_steps,
                seeds,
                override_prompt
            ],
            outputs=[output_image, mask_image],
        )

def main(host, port, debug=True, enable_queue=True):
    # initialize model
    model = VirtualStagingModel(use_cuda=True)
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Tab("Virtual Staging"):
                interface(model.generate_image)
    app.launch(debug=debug, enable_queue=enable_queue,
               server_port=port, server_name=host)



if __name__=='__main__':
    port=3105
    host='0.0.0.0'
    main(host, port)
