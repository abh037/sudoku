import cv2
import numpy as np
from math import atan2
from network import Net, train
import torch
import torchvision.transforms as transforms
from PIL import Image

def is_invertible(a):
    """
    Checks invertability
    See https://stackoverflow.com/a/17931970
    """
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    if is_invertible(A):
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return (x0, y0)
    else:
        return None

def filter(lines):

    """
    Filters similar Hough lines
    See https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python
    """
    rho_threshold = 45
    theta_threshold = 0.1

    # how many lines are similar to a given one
    similar_lines = {i : [] for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue

            rho_i,theta_i = lines[i][0]
            rho_j,theta_j = lines[j][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                similar_lines[i].append(j)

    # ordering the INDECES of the lines by how many are similar to them
    indices = [i for i in range(len(lines))]
    indices.sort(key=lambda x : len(similar_lines[x]))

    # line flags is the base for the filtering
    line_flags = len(lines)*[True]
    for i in range(len(lines) - 1):
        if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
            continue

        for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
            if not line_flags[indices[j]]: # and only if we have not disregarded them already
                continue

            rho_i,theta_i = lines[indices[i]][0]
            rho_j,theta_j = lines[indices[j]][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now
    filtered_lines = []

    for i in range(len(lines)): # filtering
        if line_flags[i]:
            filtered_lines.append(lines[i])
    return filtered_lines


def preprocess(path):
    img = path
    height, width, _ = img.shape
    resize_scaling = 800 / width
    resize_width = int(width * resize_scaling)
    resize_hieght = int(height * resize_scaling)
    resized_dimensions = (resize_width, resize_hieght)
    img = cv2.resize(img, resized_dimensions, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
      
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) < 4):
        return None

    max_area = 0
    c = 0
    for i in contours:
            area = cv2.contourArea(i)
            if area > max_area:
                max_area = area
                best_cnt = i
            c+=1


    mask = np.zeros((img.shape),np.uint8)
    cv2.drawContours(mask,[best_cnt],0,(255, 255, 255),-1)
    cv2.drawContours(mask,[best_cnt],0,(0, 0, 0),2)
    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]
    img = out
    gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray,10,31,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)[1:]
    
    x = np.mean([c[0] for c in corners])
    y = np.mean([c[1] for c in corners])
    corners = np.array(rotational_sort(corners, (x,y), True))
    if len(corners) != 4:
        return None
    return mask, corners, img
    
def argsort(seq):
    """
    "http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    by unutbu
    https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
    from Boris Gorelik
    """
    return sorted(range(len(seq)), key=seq.__getitem__)

def rotational_sort(list_of_xy_coords, centre_of_rotation_xy_coord, clockwise=True):
    """
    https://stackoverflow.com/a/67735985
    """
    cx,cy=centre_of_rotation_xy_coord
    angles = [atan2(x-cx, y-cy) for x,y in list_of_xy_coords]
    indices = argsort(angles)
    if clockwise:
        return [list_of_xy_coords[i] for i in indices]
    else:
        return [list_of_xy_coords[i] for i in indices[::-1]]
        
def warp(corners, img):
    #cv2.imwrite('board.jpg', img)
    blen = int(cv2.norm(corners[0] - corners[1] + 5, cv2.NORM_L2))
    board = np.zeros(blen)
    dst = np.float32([[0, 0], [0, blen], [blen, blen], [blen, 0]])
    try:
        M = cv2.getPerspectiveTransform(corners, dst)
    except cv2.error:
        return None
    return  cv2.warpPerspective(img, M, (blen, blen))
    
def getNumber(image):
    imsize = 256
    loader = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
    def image_loader(image):
        image = loader(image).float()
        image = torch.autograd.Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        return image
        
    
    net = Net()
    net.load_state_dict(torch.load('cnet.p'))
    net.eval()
    guess = net(image_loader(image)).detach().numpy()[0]
    number = np.argmax(guess)
    return number
    

def getBoard(path):
    x = preprocess(path)
    if x is None:
        return None
    _, corners, img = x
    board = warp(corners, img)
    if board is None:
        return None
    
    board = cv2.resize(board, (450, 450), interpolation=cv2.INTER_AREA)
    return board

def getAnsString(sols, board):
    string = ""
    if len(sols) < 8 or len(board) < 8:
        return string
    for i in range(9):
        for j in range(9):
            s = sols[i][j]
            b = board[i][j]
            string += str(" " if s == 0 or s == b else str(s))
        string += "\n"
    return string
    
def getBoardArray(path):
    board = getBoard(path)
    array = []
    if board is None:
        return array
    board = cv2.bitwise_not(board)
    board = cv2.cvtColor(board,cv2.COLOR_BGR2GRAY)
    
    board = cv2.adaptiveThreshold(board,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,49,-20)
    cropsize = 2
    #cv2.imwrite('board.jpg', board)
    for i in range(9):
        temp = []
        for j in range(9):
            x = i*50
            y = j*50
            tile = board[x:x+50, y:y+50]
            length, height = tile.shape
            tile = tile[cropsize:length-cropsize, cropsize:height-cropsize]
            tile = cv2.resize(tile, (28, 28), interpolation=cv2.INTER_AREA)
            tile = np.stack((tile,)*3, axis=-1)
            #cv2.imwrite(str(10*i+j)+'.jpg', tile)
            temp.append(getNumber(tile))
        array.append(temp)
    return array

def isValidBoard(arr):
    ret = True
    ct = 0
    flat_list = [item for sublist in arr for item in sublist]
    for i in range(1, 9):
        n = flat_list.count(i)
        if n > 10:
            ret = False
        ct += n
    
    return len(arr) == 9 and ct > 15 and ret
