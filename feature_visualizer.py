import os

import matplotlib.pyplot as plt
import torch.utils.tensorboard
import torchvision
import tqdm

import lenet


def get_feature_maps(feature_maps: dict, name: str):
    def hook(model, input, output):
        feature_maps[name] = output.detach()
    return hook


if __name__ == '__main__':
    # 1. Dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ])
    testset = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transform)

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = lenet.LeNet().to(device)
    model_name = model.__str__().split('(')[0]
    if os.path.exists('weights/{}_best.pth'.format(model.__str__().split('(')[0])):
        model.load_state_dict(torch.load('weights/{}_best.pth'.format(model.__str__().split('(')[0])))
    else:
        print('FileNotFound: pretrained_weights (' + model_name + ')')

    # 이미지 불러오기
    image_number = input('Enter the image number of the dataset>>> ')
    if image_number == '':
        image_number = 0
    else:
        image_number = int(image_number)
    image, _ = testset[image_number]
    image = image.unsqueeze(0).to(device)

    # 모델의 각 계층에 특징맵을 받아오는 hook을 등록
    feature_maps = {}
    if model_name == 'LeNet':
        model.conv1.register_forward_hook(get_feature_maps(feature_maps, 'conv1'))
        model.conv2.register_forward_hook(get_feature_maps(feature_maps, 'conv2'))
    elif model_name == 'CustomLeNet':
        model.conv1.register_forward_hook(get_feature_maps(feature_maps, 'conv1'))
        model.bn1.register_forward_hook(get_feature_maps(feature_maps, 'bn1'))
        model.conv2.register_forward_hook(get_feature_maps(feature_maps, 'conv2'))
        model.bn2.register_forward_hook(get_feature_maps(feature_maps, 'bn2'))
    else:
        raise NameError('Wrong model')

    # 예측
    with torch.no_grad():
        output = model(image)

    # 각 계층의 feature maps 저장
    for layer in tqdm.tqdm(feature_maps.keys(), desc='Saving'):
        result_dir = os.path.join('feature_maps', model_name, layer)
        os.makedirs(result_dir, exist_ok=True)
        feature_map = feature_maps[layer].squeeze().cpu()

        for i in tqdm.tqdm(range(feature_map.size()[0]), desc='Channels', leave=False):
            plt.imsave(os.path.join(result_dir, '{}.png'.format(i + 1)), feature_map[i])
