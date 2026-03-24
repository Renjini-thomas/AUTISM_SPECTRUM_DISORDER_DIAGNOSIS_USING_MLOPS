import gradio as gr
import tempfile
import shutil
import os
import numpy as np

from src.prediction.ASD_prediction import ASD_Prediction

pipeline = ASD_Prediction()


def predict_asd(file):

    if file is None:
        return "Upload MRI", None

    # ⭐ keep original extension
    ext = os.path.splitext(file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:

        shutil.copyfile(file.name, tmp.name)

        pred, prob ,slices= pipeline.predict(tmp.name)

    autism_prob = float(
        prob[list(pipeline.model.classes_).index("autism")]
    )
    gallery = [np.uint8(s) for s in slices]

    return pred, autism_prob, gallery


demo = gr.Interface(
    fn=predict_asd,
    inputs=gr.File(
        label="Upload MRI File",
        file_types=[".mgz", ".nii", ".nii.gz"]
    ),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Autism Probability"),
        gr.Gallery(label="MRI Slices")
    ],
    title="Autism Spectrum Disorder MRI Prediction",
    description="Upload MRI scan (.mgz / .nii / .nii.gz)"
)

demo.launch()