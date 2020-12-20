import numpy as np

"""
Different Methods of Combining Arrays
"""

arr1 = np.arange(1,17).reshape(4,4)
arr2 = np.arange(101,117).reshape(4,4)

arr_all1 = np.stack((arr1, arr2))
arr_all2 = np.vstack((arr1, arr2))
arr_all3 = np.hstack((arr1, arr2))
arr_all4 = np.dstack((arr1, arr2))  #Stack arrays in sequence depth wise (along third axis).
arr_all5 = np.concatenate((arr1, arr2), axis=0)
arr_all6 = np.concatenate((arr1, arr2), axis=1)

print(arr1)
print(arr2)
print("******************stack********************")
print(arr_all1)
print("*****************vstack********************")
print(arr_all2)
print("*****************hstack********************")
print(arr_all3)
print("*****************dstack********************")
print(arr_all3)
print("*************concate(axis=0)********************")
print(arr_all4)
print("*************concate(axis=1)********************")
print(arr_all5)