import matplotlib.pyplot as plt
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from PIL import Image


class FusionEnsemble_nlists(nn.Module):
    def __init__(self, model_lists, device='cpu'):
        lists = []
        for _ in model_lists:
            _model = ptcv_get_model(_, pretrained=True).to(device)
            lists.append(_model)
        self.models = nn.ModuleList(lists)

    def forward(self, x):
        y = []
        for _model in self.models:
            y.append(_model(x.clone()))

        res = None
        for _y in y:
            if res is None:
                res = _y
            else:
                res += _y

        res /= len(self.models)

        return res


class Test(object):
    def __init__(self, eval_model, classes, transform, device, file):
        self.eval_model = eval_model
        self.classes = classes
        self.transform = transform
        self.device = device
        self.file = file

        self.eval_models = [FusionEnsemble_nlists(
            [_], device=self.device) for _ in self.eval_model]

    def test(self):
        for eval_model_name, eval_model in zip(self.eval_model, self.eval_models):
            plt.figure(figsize=(10, 20))
            cnt = 0
            diff_dict = {}
            acc_old = {}
            acc_new = {}

            for i, cls_name in enumerate(self.classes):
                diff_dict[cls_name] = 0
                acc_old[cls_name] = 0
                acc_new[cls_name] = 0

                for _ in range(1, 21):
                    path = f'{cls_name}/{cls_name}{_}.png'

                    # benign image
                    im = Image.open(f'./data/{path}')
                    logit = eval_model(
                        self.transform(im).unsqueeze(0).to(self.device))[0]
                    predict_1 = logit.argmax(-1).item()
                    prob = logit.softmax(-1)[predict_1].item()

                    if _ == 1:
                        cnt += 1
                        '''plt.subplot(len(self.classes), 4, cnt)
                        plt.title(
                            f'benign: {cls_name}1.png\n{self.classes[predict_1]}: {prob:.2%}')
                        plt.axis('off')
                        plt.imshow(np.array(im))'''

                    # adversarial image
                    im = Image.open(f'./{self.file}/{path}')
                    logit = eval_model(
                        self.transform(im).unsqueeze(0).to(self.device))[0]
                    predict_2 = logit.argmax(-1).item()
                    prob = logit.softmax(-1)[predict_2].item()

                    if _ == 1:
                        cnt += 1
                        '''plt.subplot(len(self.classes), 4, cnt)
                        plt.title(
                            f'adversarial: {cls_name}1.png\n{self.classes[predict_2]}: {prob:.2%}')
                        plt.axis('off')
                        plt.imshow(np.array(im))'''

                    if predict_1 == i:
                        acc_old[cls_name] += 1

                    if predict_2 == i:
                        acc_new[cls_name] += 1

                    if predict_1 != predict_2:
                        diff_dict[cls_name] += 1

            diff_tot = 0
            acc_old_tot = 0
            acc_new_tot = 0

            for k in self.classes:
                diff_tot += diff_dict[k]
                acc_old_tot += acc_old[k]
                acc_new_tot += acc_new[k]

            print("model_name: {}".format(eval_model_name))
            print("diff_tot: {}/200".format(diff_tot))
            print("acc_old_tot: {}/200".format(acc_old_tot))
            print("acc_new_tot: {}/200".format(acc_new_tot))
            for k in self.classes:
                print("cls {}: diff_tot: {}/20 acc_old: {}/20 acc_new: {}/20".format(k,
                      diff_dict[k], acc_old[k], acc_new[k]))

            '''plt.tight_layout()
            plt.show()'''
            print("-" * 50)
