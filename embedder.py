from InstructorEmbedding import INSTRUCTOR
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
def get_embedding_model():
    model = INSTRUCTOR("hkunlp/instructor-base")
    model = model.to("cuda")
    return model
