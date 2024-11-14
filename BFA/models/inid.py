import torch.nn as nn

__all__ = ['inid']


def inid(pretrained=True, **kwargs):
    model = SimpleModel()
    return model


class SimpleModel(nn.Module):
    def __init__(self, input_size=83, output_size=3):
        super(SimpleModel, self).__init__()

        # Định nghĩa các lớp trong mô hình
        self.fc1 = nn.Linear(input_size, 128)  # Lớp đầu tiên với 128 nơ-ron
        self.fc2 = nn.Linear(128, 64)  # Lớp thứ hai với 64 nơ-ron
        self.fc3 = nn.Linear(64, output_size)  # Lớp đầu ra với 3 nơ-ron

        # Hàm kích hoạt
        self.relu = nn.ReLU()

    def forward(self, x):
        # Định nghĩa quá trình truyền dữ liệu qua các lớp
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Lớp đầu ra không sử dụng hàm kích hoạt (softmax hoặc sigmoid tùy vào bài toán)
        return x
