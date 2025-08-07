import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.current_device())  # Should return the current device index
print(torch.cuda.get_device_name())  # Should return the name of the first GPU