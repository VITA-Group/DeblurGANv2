import os
import cv2

# # image path
# im_dir = 'D:\github repo\DeblurGANv2\submit'
# # output video path
# video_dir = 'D:\github repo\DeblurGANv2'
# if not os.path.exists(video_dir):
#     os.makedirs(video_dir)
# # set saved fps
# fps = 20
# # get frames list
# frames = sorted(os.listdir(im_dir))
# # w,h of image
# img = cv2.imread(os.path.join(im_dir, frames[0]))
# #
# img_size = (img.shape[1], img.shape[0])
# # get seq name
# seq_name = os.path.dirname(im_dir).split('/')[-1]
# # splice video_dir
# video_dir = os.path.join(video_dir, seq_name + '.avi')
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# # also can write like:fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# # if want to write .mp4 file, use 'MP4V'
# videowriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
#
# for frame in frames:
#     f_path = os.path.join(im_dir, frame)
#     image = cv2.imread(f_path)
#     videowriter.write(image)
#     print(frame + " has been written!")
#
# videowriter.release()


# def main():
#     data_path = 'D:\github repo\DeblurGANv2\submit'
#     fps = 30  # 视频帧率
#     size = (1280, 720)  # 需要转为视频的图片的尺寸
#     video = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
#
#     for i in range(1029):
#         image_path = data_path + "%010d_color_labels.png" % (i + 1)
#         print(image_path)
#         img = cv2.imread(image_path)
#         video.write(img)
#         video.release()
#
#
#
#
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()


import cv2
import os
import numpy as np
from PIL import Image


def frame2video(im_dir, video_dir, fps):
    im_list = os.listdir(im_dir)
    im_list.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))  # 最好再看看图片顺序对不
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size  # 获得图片分辨率，im_dir文件夹下的图片分辨率需要一致

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G') #opencv版本是2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv版本是3
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    # count = 1
    for i in im_list:
        im_name = os.path.join(im_dir + i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
        # count+=1
        # if (count == 200):
        #     print(im_name)
        #     break
    videoWriter.release()
    print('finish')


if __name__ == '__main__':
    # im_dir = 'D:\github repo\DeblurGANv2\submit\\'  # 帧存放路径
    im_dir = 'D:\github repo\DeblurGANv2\dataset1\\blur\\'  # 帧存放路径\\'  # 帧存放路径
    video_dir = 'D:\github repo\DeblurGANv2/test.avi'  # 合成视频存放的路径
    fps = 30  # 帧率，每秒钟帧数越多，所显示的动作就会越流畅
    frame2video(im_dir, video_dir, fps)