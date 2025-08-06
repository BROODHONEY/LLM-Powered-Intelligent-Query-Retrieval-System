from InstructorEmbedding import INSTRUCTOR

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = INSTRUCTOR("hkunlp/instructor-base")
    return _model
