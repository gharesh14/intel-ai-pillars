#importing the required packages
import PIL.Image as Image
import numpy as np
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

#function for depth estimation
def depth():

    # Load the model and processor
    image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")

    # Open the image
    image = Image.open('./images/classroom1.jpg')

    # Prepare the image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to the original image size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Convert the prediction to a NumPy array and scale it
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")

    # Create an image from the array and display it
    depth = Image.fromarray(formatted)
    depth.show()

    # Save the depth map image
    depth.save('./images/depth1.jpg')  # Save as a PNG file