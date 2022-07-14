# import os
# def get_files():
#     list=[]
#     for filepath,dirnames,filenames in os.walk(r'.\dataset1\blur'):
#         for filename in filenames:
#             list.append(os.path.join(filepath,filename))
#     return list
#
# a=get_files()
# print(len(a))
#
# # for i in a:
# #     print(i)
import cv2
print(cv2.__version__)