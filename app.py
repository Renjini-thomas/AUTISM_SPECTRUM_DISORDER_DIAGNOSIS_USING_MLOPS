import gradio as gr
import tempfile
import shutil
import os
import numpy as np

from src.prediction.ASD_prediction import ASD_Prediction

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = ASD_Prediction()
    return pipeline


def predict_asd(file):

    if file is None:
        return "Upload MRI", None, None

    ext = os.path.splitext(file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:

        shutil.copyfile(file.name, tmp.name)
        pipeline = get_pipeline()

        pred, prob, slices = pipeline.predict(tmp.name)

    autism_prob = float(
        prob[list(pipeline.model.classes_).index("autism")]
    )

    gallery = [np.uint8(s) for s in slices]

    return pred, autism_prob, gallery


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🧠 Autism Spectrum Disorder MRI Prediction System  
        Upload a structural MRI scan to predict ASD risk using ML pipeline
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            file_input = gr.File(
                label="📂 Upload MRI File",
                file_types=[".mgz", ".nii", ".nii.gz"]
            )

            predict_btn = gr.Button("🚀 Run Prediction")

        with gr.Column(scale=1):

            pred_box = gr.Textbox(label="🧾 Prediction")

            prob_box = gr.Number(label="📊 Autism Probability")

    slice_gallery = gr.Gallery(
        label="🧩 Extracted Sagittal Brain Slices",
        columns=5,
        height=300
    )

    predict_btn.click(
        fn=predict_asd,
        inputs=file_input,
        outputs=[pred_box, prob_box, slice_gallery]
    )


demo.launch()