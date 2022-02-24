from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn as nn


class FusionEnsemble(nn.Module):
    def __init__(self, model_lists, device='cpu'):
        super(FusionEnsemble, self).__init__()

        self.model1 = ptcv_get_model(
            model_lists[0], pretrained=True).to(device)
        self.model2 = ptcv_get_model(
            model_lists[1], pretrained=True).to(device)
        self.model3 = ptcv_get_model(
            model_lists[2], pretrained=True).to(device)
        self.model4 = ptcv_get_model(
            model_lists[3], pretrained=True).to(device)
        self.model5 = ptcv_get_model(
            model_lists[4], pretrained=True).to(device)
        self.model6 = ptcv_get_model(
            model_lists[5], pretrained=True).to(device)
        self.model7 = ptcv_get_model(
            model_lists[6], pretrained=True).to(device)
        self.model8 = ptcv_get_model(
            model_lists[7], pretrained=True).to(device)
        self.model9 = ptcv_get_model(
            model_lists[8], pretrained=True).to(device)
        self.model10 = ptcv_get_model(
            model_lists[9], pretrained=True).to(device)
        self.model11 = ptcv_get_model(
            model_lists[10], pretrained=True).to(device)
        self.model12 = ptcv_get_model(
            model_lists[11], pretrained=True).to(device)
        self.model13 = ptcv_get_model(
            model_lists[12], pretrained=True).to(device)
        self.model14 = ptcv_get_model(
            model_lists[13], pretrained=True).to(device)
        self.model15 = ptcv_get_model(
            model_lists[14], pretrained=True).to(device)

    def forward(self, x):
        x1 = self.model1(x.clone())
        x2 = self.model2(x.clone())
        x3 = self.model3(x.clone())
        x4 = self.model4(x.clone())
        x5 = self.model5(x.clone())
        x6 = self.model6(x.clone())
        x7 = self.model7(x.clone())
        x8 = self.model8(x.clone())
        x9 = self.model9(x.clone())
        x10 = self.model10(x.clone())
        x11 = self.model11(x.clone())
        x12 = self.model12(x.clone())
        x13 = self.model13(x.clone())
        x14 = self.model14(x.clone())
        x15 = self.model15(x.clone())

        x = (x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15) / 15

        return x
