import torch
import numpy as np
import cv2

NODE_NAME = 'FaceSimilarity'

def tensor2cv2(image:torch.Tensor) -> np.array:
    if image.dim()==4:
        image = image.squeeze()
    npimage = image.numpy()
    cv2image = np.uint8(npimage * 255 / npimage.max())
    return cv2.cvtColor(cv2image, cv2.COLOR_RGB2BGR)

class FaceSimilarity:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['face_recognition']
        return {
            "required": {
                "image1": ("IMAGE", ),  #
                "image2": ("IMAGE",),  #
                "detect_method": (detect_mode,),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("similarity",)
    FUNCTION = 'face_similarity'
    CATEGORY = '😺dzNodes/FaceSimilarity'
    OUTPUT_NODE = True

    def face_similarity(self, image1, image2, detect_method
                  ):

        cvimage1 = tensor2cv2(image1)
        cvimage2 = tensor2cv2(image2)
        if detect_method == 'face_recognition':
            import  face_recognition
            face1 = face_recognition.face_locations(cvimage1)
            face2 = face_recognition.face_locations(cvimage2)
            face_encoder1 = face_recognition.face_encodings(cvimage1, face1)[0]
            face_encoder2 = face_recognition.face_encodings(cvimage2, face2)[0]
            similarity = face_recognition.face_distance([face_encoder1], face_encoder2)[0]

        similarity = (1 - similarity) * 100
        similarity = round(similarity, 2)
        return (similarity,)

NODE_CLASS_MAPPINGS = {
    "Face Similarity": FaceSimilarity
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Face Similarity": "Face Similarity"
}