#coding:utf-8
from skimage import io, color
from PIL import Image as image #加载PIL包，用于加载创建图片



def getMockResult(filename):
    fr = open(filename)
    arraylines = fr.readlines()
    MockResult = []
    for line in arraylines:
        linestr = line.strip()  # 不加任何参数去除的是空格
        linestrlist = linestr.split(' ')  # 用split会返回一个list
        MockResult.append(linestrlist[-1])
    return MockResult

if __name__ == '__main__':
    rgb = io.imread('../slic_segment/9004581.jpg')
    lab_arr = color.rgb2lab(rgb)
    pic_new = image.new("L", (lab_arr.shape[1], lab_arr.shape[0]))

    fo = open('clusters9004581.txt')
    clusterslines = fo.readlines()



    mocklabel = getMockResult('9004581-3-31.solution')
    clusterArrList = []
    clusterArrList.extend(list(set(mocklabel)))
    for label in clusterArrList:
        index = 0
        for element in mocklabel:
            if element == label:
                clusterslinelist = clusterslines[index].split('s')
                if clusterslinelist[0] == '\n':
                    print 111
                    continue
                for pairlen in range(len(clusterslinelist)-1):
                    pairElement = clusterslinelist[pairlen].split(',')
                    p1 = int(pairElement[1].replace(')',''))
                    p0 = int(pairElement[0].replace('(',''))
                    pic_new.putpixel((p1, p0), int(256 / (int(label) + 1)))
            index += 1

    pic_new.save("../mockimage/3000323-(random)-1000-10-Multi.jpg", "JPEG")

