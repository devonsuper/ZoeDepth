import torch

from zoedepth.utils.misc import get_image_from_url
from zoedepth.utils.misc import pil_to_batched_tensor

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

repo = "isl-org/ZoeDepth"
# # Zoe_N
# model = torch.hub.load(repo, "ZoeD_N", pretrained=True, force_reload=True).to("cuda")

# conf = get_config("zoedepth", "infer")
# model_zoe_n = build_model(conf)

URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS4W8H_Nxk_rs3Vje_zj6mglPOH7bnPhQitBH8WkqjlqQVotdtDEG37BsnGofME3_u6lDk&usqp=CAU"
image = get_image_from_url(URL)
x = pil_to_batched_tensor(image).numpy()#.tolist()

#print("input shape: ", x.shape)
#for i in range(x.shape[0]):
#    x[i] = x[i].tolist()

with torch.no_grad():
    # model = torch.hub.load(repo, "ZoeD_N", pretrained=True, force_reload=True).to("cuda")

    conf = get_config("zoedepth", "infer")
    model= build_model(conf)

    

    input_shape = (1, 3, 193, 260)#tuple(next(model.parameters()).size())
    x = torch.ones(input_shape)

    model(x)
    # model.printstuff()


    #torch.save(model, "zoedepth.pt")

    torch.onnx.export(model, x, "zoe_depth.onnx", opset_version=13, verbose=True)
