from lib import *

# Cố định các chỉ số random để không thay đổi kết quả khi sử dụng vào project khác
torch.manual_seed(1234)     
np.random.seed(1234)
random.seed(1234)
# 2 thông số chạy trên GPU để cố định chạy nhiều lần cũng giống nhau
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Nếu cài đặt giữ kết quả như trên thì thời gian train sẽ lâu hơn so với train lại từ đầu

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

batch_size = 4      # 1 lần train đc 4 ảnh

num_epochs = 2

save_path = './weight_fine_tuning.pth'

class_index = ["ants", "bees"]
