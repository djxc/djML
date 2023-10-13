
import sys
from pathlib import Path

currentPath = sys.path[0]
parentPath = Path(currentPath).parent
sys.path.append(str(parentPath))
sys.path.append(str(parentPath.parent))
from image_noise.remove_noise_fft import RemoveNosiseFFT

if __name__ == "__main__":
    tif_path = r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230310T191324_20230313T171720\GS18_MSS_L0S_20230310T191324_20230313T171720_B01_trans_rm.tif"
    # tif_path = r"D:\竖条纹图像.png"
    remove_noise = RemoveNosiseFFT()
    # remove_noise.draw_img_ms(tif_path)
    # remove_noise.remove_vertical_nosize(tif_path)
    remove_noise.remove_horizontal_nosize(tif_path)
    # remove_noise.remove_vh_noise(tif_path)